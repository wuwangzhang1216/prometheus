#!/usr/bin/env python3
"""A/B comparison: baseline vs quality-optimised abliteration on Qwen3.5-0.8B.

Runs a minimal but representative experiment comparing:
  A) Baseline: mean + orthogonal_projection (current Prometheus default)
  B) Quality:  COSMIC + projected_abliteration + discriminative layers + winsorize

Measures refusal count and KL divergence for both, prints a side-by-side table.
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

console = Console()

MODEL_ID = "Qwen/Qwen3-0.6B"
N_TRAIN = 200        # prompts for steering vector computation
N_EVAL = 100         # prompts for evaluation
MAX_GEN_TOKENS = 50
BATCH_SIZE = 8
STRENGTH = 1.0       # fixed steering strength (skip Optuna for speed)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_datasets(config):
    from abliterix.data import load_prompt_dataset
    console.print("Loading datasets...")
    benign_train = load_prompt_dataset(config, config.benign_prompts)
    target_train = load_prompt_dataset(config, config.target_prompts)
    benign_eval = load_prompt_dataset(config, config.benign_eval_prompts)
    target_eval = load_prompt_dataset(config, config.target_eval_prompts)
    console.print(
        f"  train: {len(benign_train)} benign, {len(target_train)} target | "
        f"eval: {len(benign_eval)} benign, {len(target_eval)} target"
    )
    return benign_train, target_train, benign_eval, target_eval


def extract_states(engine, messages, label=""):
    console.print(f"  Extracting residuals for {label}...")
    return engine.extract_hidden_states_batched(messages)


def compute_kl(engine, benign_eval):
    """KL divergence between steered and baseline next-token distributions."""
    from abliterix.util import chunk_batches

    all_kl = []
    for batch in chunk_batches(benign_eval, BATCH_SIZE):
        logprobs = engine.compute_logprobs(batch)
        # KL vs uniform is not meaningful; we compare vs stored baseline.
        all_kl.append(logprobs)
    return torch.cat(all_kl, dim=0)


def count_refusals(engine, target_eval, markers):
    """Count how many responses contain refusal markers."""
    from abliterix.util import chunk_batches

    responses = []
    for batch in chunk_batches(target_eval, BATCH_SIZE):
        responses.extend(engine.generate_text(batch))

    refusals = 0
    for resp in responses:
        lower = resp.lower()
        if any(m in lower for m in markers):
            refusals += 1
    return refusals, responses


def apply_fixed_steering(engine, vectors, config, benign_states=None, target_states=None):
    """Apply steering at a fixed strength across all components."""
    from abliterix.core.steering import apply_steering
    from abliterix.types import SteeringProfile

    n_layers = len(engine.transformer_layers)
    mid = int(0.7 * n_layers)

    profiles = {}
    for comp in engine.list_steerable_components():
        profiles[comp] = SteeringProfile(
            max_weight=STRENGTH,
            max_weight_position=float(mid),
            min_weight=0.0,
            min_weight_distance=float(n_layers * 0.5),
        )

    apply_steering(
        engine, vectors, None, profiles, config,
        benign_states=benign_states,
        target_states=target_states,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.set_grad_enabled(False)

    from abliterix.settings import AbliterixConfig
    from abliterix.types import VectorMethod
    from abliterix.vectors import compute_steering_vectors
    from abliterix.core.engine import SteeringEngine

    # Build minimal config for Qwen3-0.6B
    config = AbliterixConfig(
        model={"model_id": MODEL_ID},
        inference={"batch_size": BATCH_SIZE, "max_gen_tokens": MAX_GEN_TOKENS},
        steering={"vector_method": "mean", "orthogonal_projection": True},
        benign_prompts={
            "dataset": "datasets/good_1000",
            "split": f"train[:{N_TRAIN}]",
            "column": "prompt",
        },
        target_prompts={
            "dataset": "datasets/harmful_1000",
            "split": f"train[:{N_TRAIN}]",
            "column": "prompt",
        },
        benign_eval_prompts={
            "dataset": "datasets/good_1000",
            "split": f"train[{N_TRAIN}:{N_TRAIN + N_EVAL}]",
            "column": "prompt",
        },
        target_eval_prompts={
            "dataset": "datasets/harmful_1000",
            "split": f"train[{N_TRAIN}:{N_TRAIN + N_EVAL}]",
            "column": "prompt",
        },
    )

    markers = config.detection.compliance_markers

    # ── Load model ───────────────────────────────────────────────────────
    console.rule("[bold cyan]Loading model")
    engine = SteeringEngine(config)

    benign_train, target_train, benign_eval, target_eval = load_datasets(config)

    # ── Extract residuals (shared) ───────────────────────────────────────
    console.rule("[bold cyan]Extracting residuals")
    benign_states = extract_states(engine, benign_train, "benign")
    target_states = extract_states(engine, target_train, "target")

    # ── Baseline logprobs ────────────────────────────────────────────────
    console.rule("[bold cyan]Baseline logprobs")
    console.print("  Computing baseline next-token logprobs...")
    from abliterix.util import chunk_batches
    baseline_logprobs_parts = []
    for batch in chunk_batches(benign_eval, BATCH_SIZE):
        baseline_logprobs_parts.append(engine.compute_logprobs(batch))
    baseline_logprobs = torch.cat(baseline_logprobs_parts, dim=0)

    # ── Results storage ──────────────────────────────────────────────────
    results = {}

    # ====================================================================
    # Experiment A: Baseline (mean + orthogonal_projection)
    # ====================================================================
    console.rule("[bold yellow]Experiment A: Baseline")

    config_a = config.model_copy()
    config_a.steering.vector_method = VectorMethod.MEAN
    config_a.steering.orthogonal_projection = True
    config_a.steering.projected_abliteration = False
    config_a.steering.discriminative_layer_selection = False
    config_a.steering.winsorize_vectors = False

    console.print("  Computing steering vectors (mean + orthogonal)...")
    t0 = time.perf_counter()
    vectors_a = compute_steering_vectors(
        benign_states, target_states,
        VectorMethod.MEAN, True,
    )
    t_vec_a = time.perf_counter() - t0

    console.print("  Applying steering...")
    engine.restore_baseline()
    apply_fixed_steering(engine, vectors_a, config_a)

    console.print("  Measuring KL divergence...")
    steered_logprobs_parts = []
    for batch in chunk_batches(benign_eval, BATCH_SIZE):
        steered_logprobs_parts.append(engine.compute_logprobs(batch))
    steered_logprobs_a = torch.cat(steered_logprobs_parts, dim=0)
    kl_a = F.kl_div(steered_logprobs_a, baseline_logprobs, log_target=True, reduction="batchmean").item()

    console.print("  Counting refusals...")
    refusals_a, _ = count_refusals(engine, target_eval, markers)
    console.print(f"  [bold]Refusals: {refusals_a}/{len(target_eval)}, KL: {kl_a:.6f}[/]")

    results["A: Baseline"] = {
        "method": "mean + orthogonal",
        "refusals": refusals_a,
        "total": len(target_eval),
        "kl": kl_a,
        "vec_time": t_vec_a,
    }

    # ====================================================================
    # Experiment B: Quality (COSMIC + projected + discriminative + winsorize)
    # ====================================================================
    console.rule("[bold green]Experiment B: Quality (all improvements)")

    config_b = config.model_copy()
    config_b.steering.vector_method = VectorMethod.COSMIC
    config_b.steering.orthogonal_projection = False
    config_b.steering.projected_abliteration = True
    config_b.steering.discriminative_layer_selection = True
    config_b.steering.winsorize_vectors = True
    config_b.steering.winsorize_quantile = 0.995

    console.print("  Computing steering vectors (COSMIC + projected + winsorize)...")
    t0 = time.perf_counter()
    vectors_b = compute_steering_vectors(
        benign_states, target_states,
        VectorMethod.COSMIC, False,
        projected_abliteration=True,
        winsorize=True,
        winsorize_quantile=0.995,
    )
    t_vec_b = time.perf_counter() - t0

    console.print("  Applying steering (with discriminative layer selection)...")
    engine.restore_baseline()
    apply_fixed_steering(
        engine, vectors_b, config_b,
        benign_states=benign_states,
        target_states=target_states,
    )

    console.print("  Measuring KL divergence...")
    steered_logprobs_parts = []
    for batch in chunk_batches(benign_eval, BATCH_SIZE):
        steered_logprobs_parts.append(engine.compute_logprobs(batch))
    steered_logprobs_b = torch.cat(steered_logprobs_parts, dim=0)
    kl_b = F.kl_div(steered_logprobs_b, baseline_logprobs, log_target=True, reduction="batchmean").item()

    console.print("  Counting refusals...")
    refusals_b, _ = count_refusals(engine, target_eval, markers)
    console.print(f"  [bold]Refusals: {refusals_b}/{len(target_eval)}, KL: {kl_b:.6f}[/]")

    results["B: Quality"] = {
        "method": "COSMIC + projected + disc. layers + winsorize",
        "refusals": refusals_b,
        "total": len(target_eval),
        "kl": kl_b,
        "vec_time": t_vec_b,
    }

    # ====================================================================
    # Experiment C: Projected Abliteration only (isolate its effect)
    # ====================================================================
    console.rule("[bold blue]Experiment C: Projected Abliteration only")

    console.print("  Computing steering vectors (mean + projected)...")
    t0 = time.perf_counter()
    vectors_c = compute_steering_vectors(
        benign_states, target_states,
        VectorMethod.MEAN, False,
        projected_abliteration=True,
        winsorize=True,
    )
    t_vec_c = time.perf_counter() - t0

    config_c = config.model_copy()
    config_c.steering.discriminative_layer_selection = False

    console.print("  Applying steering...")
    engine.restore_baseline()
    apply_fixed_steering(engine, vectors_c, config_c)

    console.print("  Measuring KL divergence...")
    steered_logprobs_parts = []
    for batch in chunk_batches(benign_eval, BATCH_SIZE):
        steered_logprobs_parts.append(engine.compute_logprobs(batch))
    steered_logprobs_c = torch.cat(steered_logprobs_parts, dim=0)
    kl_c = F.kl_div(steered_logprobs_c, baseline_logprobs, log_target=True, reduction="batchmean").item()

    console.print("  Counting refusals...")
    refusals_c, _ = count_refusals(engine, target_eval, markers)
    console.print(f"  [bold]Refusals: {refusals_c}/{len(target_eval)}, KL: {kl_c:.6f}[/]")

    results["C: Projected only"] = {
        "method": "mean + projected abliteration + winsorize",
        "refusals": refusals_c,
        "total": len(target_eval),
        "kl": kl_c,
        "vec_time": t_vec_c,
    }

    # ====================================================================
    # Experiment D: Discriminative layers only (isolate its effect)
    # ====================================================================
    console.rule("[bold magenta]Experiment D: Discriminative layers only")

    config_d = config.model_copy()
    config_d.steering.orthogonal_projection = True
    config_d.steering.projected_abliteration = False
    config_d.steering.discriminative_layer_selection = True
    config_d.steering.winsorize_vectors = False

    console.print("  Applying baseline vectors with discriminative layer selection...")
    engine.restore_baseline()
    apply_fixed_steering(
        engine, vectors_a, config_d,
        benign_states=benign_states,
        target_states=target_states,
    )

    console.print("  Measuring KL divergence...")
    steered_logprobs_parts = []
    for batch in chunk_batches(benign_eval, BATCH_SIZE):
        steered_logprobs_parts.append(engine.compute_logprobs(batch))
    steered_logprobs_d = torch.cat(steered_logprobs_parts, dim=0)
    kl_d = F.kl_div(steered_logprobs_d, baseline_logprobs, log_target=True, reduction="batchmean").item()

    console.print("  Counting refusals...")
    refusals_d, _ = count_refusals(engine, target_eval, markers)
    console.print(f"  [bold]Refusals: {refusals_d}/{len(target_eval)}, KL: {kl_d:.6f}[/]")

    results["D: Disc. layers only"] = {
        "method": "mean + orthogonal + discriminative layers",
        "refusals": refusals_d,
        "total": len(target_eval),
        "kl": kl_d,
        "vec_time": t_vec_a,
    }

    # ====================================================================
    # Summary
    # ====================================================================
    console.rule("[bold cyan]Results Summary")

    table = Table(title=f"A/B Test: {MODEL_ID} — Fixed strength={STRENGTH}")
    table.add_column("Experiment", style="bold")
    table.add_column("Method")
    table.add_column("Refusals", justify="right")
    table.add_column("Refusal %", justify="right")
    table.add_column("KL Div", justify="right")
    table.add_column("KL Δ vs A", justify="right")
    table.add_column("Vec Time", justify="right")

    baseline_kl = results["A: Baseline"]["kl"]

    for name, r in results.items():
        kl_delta = r["kl"] - baseline_kl
        kl_delta_str = f"{kl_delta:+.6f}"
        if kl_delta < 0:
            kl_delta_str = f"[green]{kl_delta_str}[/]"
        elif kl_delta > 0:
            kl_delta_str = f"[red]{kl_delta_str}[/]"

        refusal_pct = r["refusals"] / r["total"] * 100
        ref_style = "green" if r["refusals"] <= results["A: Baseline"]["refusals"] else "red"

        table.add_row(
            name,
            r["method"],
            f"[{ref_style}]{r['refusals']}/{r['total']}[/]",
            f"[{ref_style}]{refusal_pct:.1f}%[/]",
            f"{r['kl']:.6f}",
            kl_delta_str,
            f"{r['vec_time']:.1f}s",
        )

    console.print(table)


if __name__ == "__main__":
    main()
