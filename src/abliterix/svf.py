# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Steering Vector Fields (SVF) — learned context-dependent steering.

Implements the core idea from arxiv:2602.01654: instead of a static steering
vector, SVF learns a differentiable concept scoring function ``f(h)`` whose
gradient ``∇_h f`` defines a context-dependent steering direction at each
activation ``h``.  This makes the steering intervention adapt to the current
hidden state, enabling more precise and reliable control.

The ConceptScorer is a small MLP trained per layer to distinguish harmful
from harmless activations.  During inference, its gradient provides the
locally optimal steering direction at each token position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .util import print


class ConceptScorer(nn.Module):
    """Small MLP that scores activations on a harmful/harmless spectrum.

    Architecture: Linear → GELU → Linear → GELU → Linear → Sigmoid

    The gradient of the output with respect to the input provides the
    context-dependent steering direction.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Score activations.  Returns values in [0, 1]."""
        return self.net(x)


def train_concept_scorers(
    benign_states: Tensor,
    target_states: Tensor,
    hidden_dim: int,
    n_epochs: int = 50,
    lr: float = 1e-3,
    hidden_dim_scorer: int = 256,
) -> dict[int, ConceptScorer]:
    """Train one ConceptScorer per transformer layer.

    Benign activations are labelled 0 (low concept score); target (harmful)
    activations are labelled 1 (high concept score).  After training, the
    gradient of the scorer w.r.t. the input provides the direction that
    maximally increases the "harmful" score — which is exactly the
    context-dependent refusal direction to steer away from.

    Parameters
    ----------
    benign_states, target_states : Tensor
        Shape ``(n, layers+1, hidden_dim)``.
    hidden_dim : int
        Input dimension of each scorer.
    n_epochs : int
        Training epochs per layer.
    lr : float
        Learning rate.
    hidden_dim_scorer : int
        Hidden dimension for the scorer MLP.

    Returns
    -------
    dict[int, ConceptScorer]
        Mapping from transformer layer index (0-based) to trained scorer.
        Index 0 corresponds to the embedding layer and is excluded.
    """
    n_layers = benign_states.shape[1]
    device = benign_states.device
    scorers: dict[int, ConceptScorer] = {}

    for layer_idx in range(1, n_layers):  # Skip embedding layer (index 0).
        b = benign_states[:, layer_idx, :].float()
        t = target_states[:, layer_idx, :].float()

        # Build dataset: benign = 0, target = 1.
        X = torch.cat([b, t], dim=0)
        y = torch.cat([
            torch.zeros(b.shape[0], 1, device=device),
            torch.ones(t.shape[0], 1, device=device),
        ])

        scorer = ConceptScorer(hidden_dim, hidden_dim_scorer).to(device)
        optimizer = torch.optim.Adam(scorer.parameters(), lr=lr)

        scorer.train()
        with torch.enable_grad():
            for _epoch in range(n_epochs):
                # Shuffle.
                perm = torch.randperm(X.shape[0], device=device)
                X_shuf, y_shuf = X[perm], y[perm]

                pred = scorer(X_shuf)
                loss = F.binary_cross_entropy(pred, y_shuf)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        scorer.eval()

        # Report final accuracy for this layer.
        with torch.no_grad():
            pred_labels = (scorer(X) > 0.5).float()
            acc = (pred_labels == y).float().mean().item()

        if acc > 0.6:  # Only keep scorers that learned something useful.
            scorers[layer_idx - 1] = scorer  # Map to 0-based layer index.

    print(
        f"* {len(scorers)}/{n_layers - 1} layers with effective scorers "
        f"(accuracy > 60%)"
    )
    return scorers
