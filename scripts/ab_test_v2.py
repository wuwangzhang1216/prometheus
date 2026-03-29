#!/usr/bin/env python3
"""A/B comparison v2: grid-search strength for each method.

For each method, sweeps strength ∈ [0.5, 1.0, 1.5, 2.0] and picks the
best Pareto point (refusal=0 with lowest KL, or lowest refusal if none
reaches 0).
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

console = Console()

MODEL_ID = "Qwen/Qwen3-0.6B"
N_TRAIN = 200
N_EVAL = 100
MAX_GEN_TOKENS = 50
BATCH_SIZE = 8
STRENGTHS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


def main():
    torch.set_grad_enabled(False)

    from abliterix.settings import AbliterixConfig
    from abliterix.types import VectorMethod
    from abliterix.vectors import compute_steering_vectors
    from abliterix.core.engine import SteeringEngine
    from abliterix.core.steering import apply_steering
    from abliterix.types import SteeringProfile
    from abliterix.data import load_prompt_dataset
    from abliterix.util import chunk_batches

    config = AbliterixConfig(
        model={"model_id": MODEL_ID},
        inference={"batch_size": BATCH_SIZE, "max_gen_tokens": MAX_GEN_TOKENS},
        steering={"vector_method": "mean", "orthogonal_projection": True},
        benign_prompts={"dataset": "datasets/good_1000", "split": f"train[:{N_TRAIN}]", "column": "prompt"},
        target_prompts={"dataset": "datasets/harmful_1000", "split": f"train[:{N_TRAIN}]", "column": "prompt"},
        benign_eval_prompts={"dataset": "datasets/good_1000", "split": f"train[{N_TRAIN}:{N_TRAIN+N_EVAL}]", "column": "prompt"},
        target_eval_prompts={"dataset": "datasets/harmful_1000", "split": f"train[{N_TRAIN}:{N_TRAIN+N_EVAL}]", "column": "prompt"},
    )
    markers = config.detection.compliance_markers

    console.rule("[bold cyan]Loading model")
    engine = SteeringEngine(config)

    console.print("Loading datasets...")
    benign_train = load_prompt_dataset(config, config.benign_prompts)
    target_train = load_prompt_dataset(config, config.target_prompts)
    benign_eval = load_prompt_dataset(config, config.benign_eval_prompts)
    target_eval = load_prompt_dataset(config, config.target_eval_prompts)

    console.rule("[bold cyan]Extracting residuals")
    benign_states = engine.extract_hidden_states_batched(benign_train)
    target_states = engine.extract_hidden_states_batched(target_train)

    console.print("Computing baseline logprobs...")
    bl_parts = []
    for batch in chunk_batches(benign_eval, BATCH_SIZE):
        bl_parts.append(engine.compute_logprobs(batch))
    baseline_logprobs = torch.cat(bl_parts, dim=0)

    n_layers = len(engine.transformer_layers)
    mid = int(0.7 * n_layers)

    def make_profiles(strength):
        profiles = {}
        for comp in engine.list_steerable_components():
            profiles[comp] = SteeringProfile(
                max_weight=strength,
                max_weight_position=float(mid),
                min_weight=0.0,
                min_weight_distance=float(n_layers * 0.5),
            )
        return profiles

    def evaluate(vectors, cfg, strength, benign_s=None, target_s=None):
        engine.restore_baseline()
        apply_steering(engine, vectors, None, make_profiles(strength), cfg,
                       benign_states=benign_s, target_states=target_s)

        # KL
        parts = []
        for batch in chunk_batches(benign_eval, BATCH_SIZE):
            parts.append(engine.compute_logprobs(batch))
        steered = torch.cat(parts, dim=0)
        kl = F.kl_div(steered, baseline_logprobs, log_target=True, reduction="batchmean").item()

        # Refusals
        responses = []
        for batch in chunk_batches(target_eval, BATCH_SIZE):
            responses.extend(engine.generate_text(batch))
        refusals = sum(1 for r in responses if any(m in r.lower() for m in markers))

        return refusals, kl

    # ── Define methods ───────────────────────────────────────────────────
    methods = {}

    # A: Baseline
    console.rule("[bold yellow]Computing vectors: Baseline")
    vectors_a = compute_steering_vectors(benign_states, target_states, VectorMethod.MEAN, True)
    config_a = config.model_copy()
    methods["A: Baseline\n(mean+ortho)"] = (vectors_a, config_a, None, None)

    # B: Projected Abliteration + winsorize
    console.rule("[bold blue]Computing vectors: Projected")
    vectors_b = compute_steering_vectors(
        benign_states, target_states, VectorMethod.MEAN, False,
        projected_abliteration=True, winsorize=True,
    )
    config_b = config.model_copy()
    config_b.steering.projected_abliteration = True
    config_b.steering.discriminative_layer_selection = False
    methods["B: Projected\n(mean+proj+win)"] = (vectors_b, config_b, None, None)

    # C: Discriminative layers
    console.rule("[bold magenta]Computing vectors: Discriminative")
    config_c = config.model_copy()
    config_c.steering.discriminative_layer_selection = True
    methods["C: Disc. layers\n(mean+ortho+disc)"] = (vectors_a, config_c, benign_states, target_states)

    # D: COSMIC + projected + discriminative
    console.rule("[bold green]Computing vectors: COSMIC combo")
    vectors_d = compute_steering_vectors(
        benign_states, target_states, VectorMethod.COSMIC, False,
        projected_abliteration=True, winsorize=True,
    )
    config_d = config.model_copy()
    config_d.steering.projected_abliteration = True
    config_d.steering.discriminative_layer_selection = True
    methods["D: Full quality\n(COSMIC+proj+disc)"] = (vectors_d, config_d, benign_states, target_states)

    # E: Optimal Transport + projected
    console.rule("[bold red]Computing vectors: Optimal Transport")
    vectors_e = compute_steering_vectors(
        benign_states, target_states, VectorMethod.OPTIMAL_TRANSPORT, False,
        projected_abliteration=True, winsorize=True,
    )
    config_e = config.model_copy()
    config_e.steering.projected_abliteration = True
    config_e.steering.discriminative_layer_selection = True
    methods["E: OT combo\n(OT+proj+disc)"] = (vectors_e, config_e, benign_states, target_states)

    # ── Grid search ──────────────────────────────────────────────────────
    all_results = {}

    for method_name, (vecs, cfg, bs, ts) in methods.items():
        console.rule(f"[bold]Sweeping: {method_name}")
        best = None
        for s in STRENGTHS:
            console.print(f"  strength={s:.1f} ... ", end="")
            refusals, kl = evaluate(vecs, cfg, s, bs, ts)
            console.print(f"refusals={refusals}, KL={kl:.6f}")

            # Pareto: first priority = lowest refusal, second = lowest KL
            if best is None or refusals < best["refusals"] or (
                refusals == best["refusals"] and kl < best["kl"]
            ):
                best = {"refusals": refusals, "kl": kl, "strength": s}

        all_results[method_name] = best
        console.print(f"  → [bold]Best: strength={best['strength']}, refusals={best['refusals']}, KL={best['kl']:.6f}[/]")

    # ── Summary table ────────────────────────────────────────────────────
    console.rule("[bold cyan]Final Results (Best Pareto Point per Method)")

    table = Table(title=f"A/B Test: {MODEL_ID}")
    table.add_column("Method", style="bold")
    table.add_column("Best λ", justify="right")
    table.add_column("Refusals", justify="right")
    table.add_column("Ref %", justify="right")
    table.add_column("KL Div", justify="right")
    table.add_column("KL Δ vs A", justify="right")

    baseline_kl = all_results["A: Baseline\n(mean+ortho)"]["kl"]
    baseline_ref = all_results["A: Baseline\n(mean+ortho)"]["refusals"]

    for name, r in all_results.items():
        kl_delta = r["kl"] - baseline_kl
        ref_delta = r["refusals"] - baseline_ref

        kl_str = f"{kl_delta:+.6f}"
        kl_str = f"[green]{kl_str}[/]" if kl_delta < -0.0001 else (f"[red]{kl_str}[/]" if kl_delta > 0.0001 else kl_str)

        ref_pct = r["refusals"] / N_EVAL * 100
        ref_style = "green" if r["refusals"] <= baseline_ref else "red"

        table.add_row(
            name,
            f"{r['strength']:.1f}",
            f"[{ref_style}]{r['refusals']}/{N_EVAL}[/]",
            f"[{ref_style}]{ref_pct:.1f}%[/]",
            f"{r['kl']:.6f}",
            kl_str,
        )

    console.print(table)

    # Winner analysis
    console.print()
    best_kl_name = min(all_results, key=lambda k: all_results[k]["kl"])
    best_ref_name = min(all_results, key=lambda k: all_results[k]["refusals"])
    console.print(f"[bold green]Lowest KL:[/] {best_kl_name.split(chr(10))[0]} (KL={all_results[best_kl_name]['kl']:.6f})")
    console.print(f"[bold green]Lowest refusal:[/] {best_ref_name.split(chr(10))[0]} (refusals={all_results[best_ref_name]['refusals']})")


if __name__ == "__main__":
    main()
