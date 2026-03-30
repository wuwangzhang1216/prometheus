"""Tests for Steering Vector Fields (SVF) — learned context-dependent steering."""

import torch

from abliterix.svf import ConceptScorer, train_concept_scorers


class TestConceptScorer:
    """Tests for the ConceptScorer MLP."""

    def test_forward_shape(self):
        """Scorer should output a scalar per input."""
        scorer = ConceptScorer(input_dim=64, hidden_dim=32)
        x = torch.randn(5, 64)
        out = scorer(x)
        assert out.shape == (5, 1)

    def test_output_range(self):
        """Scorer output should be in [0, 1] (sigmoid activation)."""
        scorer = ConceptScorer(input_dim=64, hidden_dim=32)
        x = torch.randn(100, 64)
        out = scorer(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_gradient_exists(self):
        """torch.autograd.grad should produce non-zero gradients."""
        scorer = ConceptScorer(input_dim=64, hidden_dim=32)
        scorer.eval()

        with torch.enable_grad():
            x = torch.randn(3, 64, requires_grad=True)
            out = scorer(x)
            grad = torch.autograd.grad(out.sum(), x, create_graph=False)[0]

        assert grad.shape == (3, 64)
        assert grad.abs().sum() > 0, "Gradient should be non-zero"


class TestTrainConceptScorers:
    """Tests for training concept scorers."""

    def test_returns_dict_of_scorers(self, synthetic_states):
        benign, target = synthetic_states
        scorers = train_concept_scorers(
            benign, target,
            hidden_dim=64,
            n_epochs=20,
            lr=1e-2,
            hidden_dim_scorer=32,
        )

        assert isinstance(scorers, dict)
        # Should have at least some effective scorers.
        assert len(scorers) > 0, "Should train at least one effective scorer"

        # All values should be ConceptScorer instances.
        for layer_idx, scorer in scorers.items():
            assert isinstance(scorer, ConceptScorer)
            assert isinstance(layer_idx, int)

    def test_scorers_separate_classes(self, synthetic_states):
        """Trained scorers should score target (harmful) higher than benign."""
        benign, target = synthetic_states
        scorers = train_concept_scorers(
            benign, target,
            hidden_dim=64,
            n_epochs=30,
            lr=1e-2,
            hidden_dim_scorer=32,
        )

        if not scorers:
            return  # Skip if no effective scorers trained.

        # Pick any scorer and check class separation.
        layer_idx = next(iter(scorers))
        scorer = scorers[layer_idx]

        with torch.no_grad():
            # layer_idx is 0-based transformer layer, states have +1 offset for embedding.
            b_scores = scorer(benign[:, layer_idx + 1, :].float()).mean()
            t_scores = scorer(target[:, layer_idx + 1, :].float()).mean()

        assert t_scores > b_scores, (
            f"Target scores should be higher: target={t_scores:.3f}, benign={b_scores:.3f}"
        )

    def test_layer_indices_valid(self, synthetic_states):
        """Scorer keys should be valid transformer layer indices."""
        benign, target = synthetic_states
        n_layers = benign.shape[1] - 1  # Exclude embedding.

        scorers = train_concept_scorers(
            benign, target,
            hidden_dim=64,
            n_epochs=20,
            lr=1e-2,
            hidden_dim_scorer=32,
        )

        for layer_idx in scorers:
            assert 0 <= layer_idx < n_layers, (
                f"Layer index {layer_idx} out of range [0, {n_layers})"
            )
