# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Steering algorithm: modify model weights via LoRA rank-1 updates.

This module implements the core steering (abliteration) procedure as a
standalone function rather than a method on the engine, keeping the algorithm
cleanly separated from model-management concerns.
"""

import math
from typing import cast

import bitsandbytes as bnb
import torch
import torch.linalg as LA
import torch.nn.functional as F
from peft.tuners.lora.layer import Linear
from torch import Tensor

from ..settings import AbliterixConfig
from ..types import DecayKernel, ExpertRoutingConfig, SteeringMode, SteeringProfile, WeightNorm

# Avoid circular import: accept the engine as a duck-typed object rather
# than importing SteeringEngine directly.  The caller is responsible for
# passing a valid engine instance.

_FP8_DTYPES = frozenset()
with __import__("contextlib").suppress(AttributeError):
    _FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})


def _dequantize_fp8_blockwise(
    weight: Tensor,
    weight_scale: Tensor,
    block_size: int = 128,
) -> Tensor:
    """Block-wise FP8 dequantization: W_real = weight_fp8 * scale_per_block."""
    out_f, in_f = weight.shape
    w = weight.to(torch.float32)
    scale = weight_scale.repeat_interleave(block_size, dim=0)[:out_f]
    scale = scale.repeat_interleave(block_size, dim=1)[:, :in_f]
    return w * scale


def _detect_discriminative_layers(
    steering_vectors: Tensor,
    benign_states: Tensor | None,
    target_states: Tensor | None,
) -> set[int]:
    """Identify layers where harmful/harmless activations project in opposite directions.

    A layer is *discriminative* if the mean projection of harmful activations
    onto the steering vector is positive while the mean projection of harmless
    activations is negative (or vice versa).  Only these layers benefit from
    steering; non-discriminative layers are skipped to avoid coherence damage.

    Based on: Selective Steering (2026) — 5.5× improvement with zero perplexity violations.

    Returns a set of discriminative layer indices (0-based transformer layer indices).
    """
    if benign_states is None or target_states is None:
        # Fall back to all layers if residuals are unavailable.
        return set(range(steering_vectors.shape[0] - 1))

    discriminative: set[int] = set()
    n_layers = min(steering_vectors.shape[0] - 1, benign_states.shape[1] - 1)

    for layer_idx in range(n_layers):
        v = steering_vectors[layer_idx + 1]  # +1 because index 0 is embedding
        b = benign_states[:, layer_idx + 1, :].float()
        t = target_states[:, layer_idx + 1, :].float()

        # Mean scalar projection onto steering direction.
        mu_benign = (b @ v.float()).mean().item()
        mu_target = (t @ v.float()).mean().item()

        # Discriminative = opposite signs.
        if mu_benign * mu_target < 0:
            discriminative.add(layer_idx)

    return discriminative


def _make_angular_hook(
    direction: Tensor,
    angle_degrees: float,
    adaptive: bool = False,
):
    """Create a forward hook that rotates activations within the steering plane.

    Implements Angular Steering (NeurIPS 2025 Spotlight):
        h_steered = h - proj_P(h) + |proj_P(h)| * [b1 b2] R_θ [1 0]^T

    Parameters
    ----------
    direction : Tensor
        Unit-normalised steering direction (hidden_dim,).
    angle_degrees : float
        Rotation angle.  ~200° = compliance, ~20° = refusal.
    adaptive : bool
        If True, only rotate activations positively aligned with the
        direction (Adaptive Angular Steering), reducing interference.
    """
    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def hook(module, input, output):
        h = output
        if isinstance(h, tuple):
            h = h[0]

        d = direction.to(h.device, dtype=h.dtype)

        # b1 = d (first basis vector of the 2D steering plane).
        # Scalar projection of h onto d.
        proj_scalar = (h @ d).unsqueeze(-1)  # (..., seq, 1)
        proj_on_d = proj_scalar * d           # component along b1

        # b2 = Gram-Schmidt orthogonal complement within the plane.
        residual = h - proj_on_d
        residual_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        b2 = residual / residual_norm

        # The 2D projection has components (proj_scalar, residual_norm).
        # Its magnitude is preserved by rotation.
        # Rotate: new_b1_coeff = cos(θ)*proj_scalar + sin(θ)*residual_norm
        #         new_b2_coeff = -sin(θ)*proj_scalar + cos(θ)*residual_norm
        new_proj_on_d = (cos_t * proj_scalar + sin_t * residual_norm) * d
        new_residual = (-sin_t * proj_scalar + cos_t * residual_norm) * b2

        # Components outside the 2D plane are preserved.
        # h = proj_on_d + residual + h_perp  →  h_perp = h - proj_on_d - residual
        # But residual = residual_norm * b2, so h_perp is everything else.
        # Since we only computed b2 from residual, there's nothing outside;
        # the full h is reconstructed as new_proj_on_d + new_residual.
        h_new = new_proj_on_d + new_residual

        if adaptive:
            mask = (proj_scalar > 0).float()
            h_new = mask * h_new + (1 - mask) * h

        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    return hook


def apply_steering(
    engine,  # SteeringEngine
    steering_vectors: Tensor,
    vector_index: float | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig | None = None,
    safety_experts: dict[int, list[tuple[int, float]]] | None = None,
    routing_config: ExpertRoutingConfig | None = None,
    benign_states: Tensor | None = None,
    target_states: Tensor | None = None,
):
    """Apply rank-1 LoRA steering to every steerable module in the model.

    Parameters
    ----------
    engine : SteeringEngine
        The loaded model wrapper (provides ``transformer_layers``,
        ``steerable_modules``, adapter access, and helper methods).
    steering_vectors : Tensor
        Per-layer vectors of shape ``(layers+1, hidden_dim)``.
    vector_index : float or None
        If not None, interpolate a global vector from two adjacent layers.
        If None, use per-layer vectors.
    profiles : dict
        Component-name → :class:`SteeringProfile` mapping.
    config : AbliterixConfig
        Top-level configuration (kernel choice, normalisation, etc.).
    safety_experts : dict, optional
        MoE profiling results used for expert-level steering.
    routing_config : ExpertRoutingConfig, optional
        Hyper-parameters for MoE expert suppression.
    benign_states : Tensor, optional
        Residual states from benign prompts, used for discriminative layer
        selection.  Shape ``(n, layers+1, hidden_dim)``.
    target_states : Tensor, optional
        Residual states from target prompts.  Shape ``(n, layers+1, hidden_dim)``.
    """
    if config is None:
        config = engine.config

    steering_mode = config.steering.steering_mode

    # --- Discriminative layer selection -----------------------------------
    discriminative_layers: set[int] | None = None
    if config.steering.discriminative_layer_selection:
        discriminative_layers = _detect_discriminative_layers(
            steering_vectors, benign_states, target_states,
        )

    # --- Resolve the global steering vector (if applicable) ---------------
    if vector_index is None:
        global_vector = None
    else:
        fractional, integral = math.modf(vector_index + 1)
        global_vector = F.normalize(
            steering_vectors[int(integral)].lerp(
                steering_vectors[int(integral) + 1],
                fractional,
            ),
            p=2,
            dim=0,
        )

    # --- Angular / Adaptive Angular steering (hook-based) -----------------
    if steering_mode in (SteeringMode.ANGULAR, SteeringMode.ADAPTIVE_ANGULAR):
        _apply_angular_steering(
            engine,
            steering_vectors,
            global_vector,
            profiles,
            config,
            discriminative_layers,
            adaptive=(steering_mode == SteeringMode.ADAPTIVE_ANGULAR),
        )
        # MoE expert steering still uses weight modification.
        if safety_experts and routing_config:
            _apply_moe_steering(engine, steering_vectors, global_vector, safety_experts, routing_config)
        return

    # --- Pre-cache steering vectors per device ----------------------------
    devices: set[torch.device] = set()
    for idx in range(len(engine.transformer_layers)):
        for mods in engine.steerable_modules(idx).values():
            for mod in mods:
                devices.add(mod.weight.device)

    sv_by_device = {d: steering_vectors.to(d) for d in devices}
    gv_by_device = (
        {d: global_vector.to(d) for d in devices} if global_vector is not None else None
    )

    # --- Per-layer, per-component steering --------------------------------
    kernel = config.steering.decay_kernel

    for layer_idx in range(len(engine.transformer_layers)):
        # Skip non-discriminative layers when the feature is enabled.
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue

        for component, modules in engine.steerable_modules(layer_idx).items():
            sp = profiles[component]

            distance = cast(float, abs(layer_idx - sp.max_weight_position))
            if distance > sp.min_weight_distance:
                continue

            # Compute interpolated weight using the configured decay kernel.
            t = distance / sp.min_weight_distance  # normalised ∈ [0, 1]
            if kernel == DecayKernel.GAUSSIAN:
                strength = sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(
                    -2.0 * t * t
                )
            elif kernel == DecayKernel.COSINE:
                strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
                    0.5 * (1.0 + math.cos(math.pi * t))
                )
            else:  # LINEAR
                strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)

            for mod in modules:
                # TODO: The module-interface assumption here is fragile — PEFT
                #       wraps modules differently per quantisation mode.
                mod = cast(Linear, mod)

                device = mod.weight.device
                if global_vector is None:
                    v = sv_by_device[device][layer_idx + 1]
                else:
                    v = gv_by_device[device]  # ty:ignore[non-subscriptable]

                # Obtain the full-precision weight matrix W.
                base_weight = cast(Tensor, mod.base_layer.weight)
                qs = getattr(base_weight, "quant_state", None)
                CB = getattr(base_weight, "CB", None)

                if qs is not None:
                    # 4-bit NF4: use cached dequantised weights when available
                    # to avoid repeated expensive dequantisation.
                    mid = id(mod)
                    if mid in engine._dequant_cache:
                        W = engine._dequant_cache[mid]
                    else:
                        W = cast(
                            Tensor,
                            bnb.functional.dequantize_4bit(  # ty:ignore[possibly-missing-attribute]
                                base_weight.data,
                                qs,
                            ).to(torch.float32),
                        )
                        engine._dequant_cache[mid] = W
                elif CB is not None:
                    # Int8 quantisation: dequantise from CB data and SCB row scales.
                    mid = id(mod)
                    if mid in engine._dequant_cache:
                        W = engine._dequant_cache[mid]
                    else:
                        SCB = base_weight.SCB  # ty:ignore[unresolved-attribute]
                        W = CB.float() * SCB.float().unsqueeze(1) / 127.0
                        engine._dequant_cache[mid] = W
                elif _FP8_DTYPES and base_weight.dtype in _FP8_DTYPES:
                    # FP8 block-quantized: dequantise using per-block weight_scale.
                    mid = id(mod)
                    if mid in engine._dequant_cache:
                        W = engine._dequant_cache[mid]
                    else:
                        weight_scale = getattr(mod.base_layer, "weight_scale", None)
                        if weight_scale is not None:
                            W = _dequantize_fp8_blockwise(
                                base_weight.data, weight_scale
                            )
                        else:
                            W = base_weight.to(torch.float32)
                        engine._dequant_cache[mid] = W
                else:
                    W = base_weight.to(torch.float32)

                W = W.view(W.shape[0], -1)

                # Optional row normalisation before computing the adapter.
                norm_mode = config.steering.weight_normalization
                if norm_mode != WeightNorm.NONE:
                    W_orig = W
                    W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                    W = F.normalize(W, p=2, dim=1)

                # Rank-1 steering: project W onto the orthogonal complement of v.
                #   lora_A  =  vᵀ W    (shape 1 × d_in)
                #   lora_B  = -λ v      (shape d_out × 1)
                lora_A = (v @ W).view(1, -1)
                lora_B = (-strength * v).view(-1, 1)

                if norm_mode == WeightNorm.PRE:
                    lora_B = W_row_norms * lora_B
                elif norm_mode == WeightNorm.FULL:
                    # Low-rank SVD approximation that preserves original row
                    # magnitudes after the rank-1 update.
                    W = W + lora_B @ lora_A
                    W = F.normalize(W, p=2, dim=1)
                    W = W * W_row_norms
                    W = W - W_orig
                    r = engine.peft_config.r
                    U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)
                    U = U[:, :r]
                    S = S[:r]
                    Vh = Vh[:, :r].T
                    sqrt_S = torch.sqrt(S)
                    lora_B = U @ torch.diag(sqrt_S)
                    lora_A = torch.diag(sqrt_S) @ Vh

                # Write the adapter weights (PEFT default adapter name).
                wA = cast(Tensor, mod.lora_A["default"].weight)
                wB = cast(Tensor, mod.lora_B["default"].weight)
                wA.data = lora_A.to(wA.dtype)
                wB.data = lora_B.to(wB.dtype)

    # --- MoE expert-level steering ----------------------------------------
    if safety_experts and routing_config:
        _apply_moe_steering(
            engine, steering_vectors, global_vector, safety_experts, routing_config,
            sv_by_device=sv_by_device, gv_by_device=gv_by_device,
        )


# ---------------------------------------------------------------------------
# Angular / Adaptive Angular steering (hook-based)
# ---------------------------------------------------------------------------


def _apply_angular_steering(
    engine,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    profiles: dict[str, SteeringProfile],
    config: AbliterixConfig,
    discriminative_layers: set[int] | None,
    adaptive: bool = False,
):
    """Register forward hooks that rotate activations toward the compliance arc.

    Each hook implements the Angular Steering rotation in a 2D subspace
    spanned by the steering direction and the activation's component
    orthogonal to it.  The rotation angle is mapped from the steering
    strength computed by the decay kernel.
    """
    kernel = config.steering.decay_kernel

    # Remove any previously registered angular hooks.
    if not hasattr(engine, "_angular_hooks"):
        engine._angular_hooks = []

    for layer_idx in range(len(engine.transformer_layers)):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue

        layer = engine.transformer_layers[layer_idx]

        # Compute effective strength from profiles (use first component).
        component = next(iter(profiles))
        sp = profiles[component]

        distance = cast(float, abs(layer_idx - sp.max_weight_position))
        if distance > sp.min_weight_distance:
            continue

        t = distance / sp.min_weight_distance
        if kernel == DecayKernel.GAUSSIAN:
            strength = sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(
                -2.0 * t * t
            )
        elif kernel == DecayKernel.COSINE:
            strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
                0.5 * (1.0 + math.cos(math.pi * t))
            )
        else:  # LINEAR
            strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)

        # Map strength to rotation angle.  strength=1.0 → 180° (full inversion).
        angle = strength * 180.0

        if global_vector is None:
            v = steering_vectors[layer_idx + 1]
        else:
            v = global_vector

        hook = _make_angular_hook(v, angle, adaptive=adaptive)
        handle = layer.register_forward_hook(hook)
        engine._angular_hooks.append(handle)


# ---------------------------------------------------------------------------
# MoE expert-level steering
# ---------------------------------------------------------------------------


def _apply_moe_steering(
    engine,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    safety_experts: dict[int, list[tuple[int, float]]],
    routing_config: ExpertRoutingConfig,
    *,
    sv_by_device: dict | None = None,
    gv_by_device: dict | None = None,
):
    """Apply router-weight suppression and fused-expert abliteration."""
    n_suppress = routing_config.n_suppress
    bias_value = routing_config.router_bias
    expert_w = routing_config.expert_ablation_weight

    # Build device caches if not provided.
    if sv_by_device is None:
        devices: set[torch.device] = set()
        for idx in range(len(engine.transformer_layers)):
            for mods in engine.steerable_modules(idx).values():
                for mod in mods:
                    devices.add(mod.weight.device)
        sv_by_device = {d: steering_vectors.to(d) for d in devices}
        gv_by_device = (
            {d: global_vector.to(d) for d in devices} if global_vector is not None else None
        )

    for layer_idx in range(len(engine.transformer_layers)):
        if layer_idx not in safety_experts:
            continue

        layer = engine.transformer_layers[layer_idx]
        top = safety_experts[layer_idx][:n_suppress]
        if not top:
            continue

        # Pick the steering vector for this layer.
        any_device = next(iter(sv_by_device))
        if global_vector is None:
            v = sv_by_device[any_device][layer_idx + 1]
        else:
            v = gv_by_device[any_device]  # ty:ignore[non-subscriptable]

        # (A) Router-weight suppression
        gate = engine._locate_router(layer)
        if gate is not None and bias_value < 0:
            scale = max(0.0, 1.0 + bias_value / 10.0)
            for eid, _ in top:
                engine._router_originals.append(
                    (layer_idx, eid, gate.weight.data[eid].clone())
                )
                gate.weight.data[eid] *= scale

        # (B) Fused-expert down-projection steering
        fused = engine._locate_fused_weights(layer)
        if fused is not None and expert_w > 0:
            v_dev = v.to(fused.device)
            # Pre-fetch FP8 scale for this fused parameter (if applicable).
            fused_scale = None
            if _FP8_DTYPES and fused.dtype in _FP8_DTYPES:
                for attr in ("weight_scale", "scale"):
                    fused_scale = getattr(fused, attr, None)
                    if fused_scale is None:
                        # Try the parent module (mlp.experts).
                        with __import__("contextlib").suppress(Exception):
                            fused_scale = getattr(layer.mlp.experts, attr, None)
                    if fused_scale is not None:
                        break
            for eid, _ in top:
                if fused_scale is not None:
                    W = _dequantize_fp8_blockwise(fused.data[eid], fused_scale)
                else:
                    W = fused.data[eid].to(torch.float32)
                vTW = v_dev.float() @ W
                W -= expert_w * torch.outer(v_dev.float(), vTW)
                fused.data[eid] = W.to(fused.dtype)

                engine._expert_deltas.append(
                    (layer_idx, eid, expert_w, v_dev.float().cpu(), vTW.cpu())
                )
