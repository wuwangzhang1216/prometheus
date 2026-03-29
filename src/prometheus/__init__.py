# Prometheus — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prometheus — Automated model steering and alignment adjustment via LoRA-based optimization."""

from .core.engine import SteeringEngine
from .eval.detector import RefusalDetector
from .eval.scorer import TrialScorer
from .settings import PrometheusConfig
from .types import (
    ChatMessage,
    DecayKernel,
    ExpertRoutingConfig,
    QuantMode,
    SteeringProfile,
    VectorMethod,
    WeightNorm,
)

__all__ = [
    "ChatMessage",
    "DecayKernel",
    "ExpertRoutingConfig",
    "PrometheusConfig",
    "QuantMode",
    "RefusalDetector",
    "SteeringEngine",
    "SteeringProfile",
    "TrialScorer",
    "VectorMethod",
    "WeightNorm",
]
