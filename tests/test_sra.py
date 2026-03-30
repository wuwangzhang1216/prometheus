"""Tests for Surgical Refusal Ablation (SRA) — concept-guided spectral cleaning."""

import torch
import torch.nn.functional as F

from abliterix.sra import _build_concept_atoms, _spectral_residualize, compute_sra_vectors
from abliterix.types import VectorMethod


class TestBuildConceptAtoms:
    """Tests for _build_concept_atoms."""

    def test_output_shape(self, synthetic_states):
        benign, _ = synthetic_states
        n_atoms = 8
        atoms = _build_concept_atoms(benign, n_atoms=n_atoms)

        # Should be (n_atoms, n_layers, hidden_dim)
        assert atoms.shape == (n_atoms, benign.shape[1], benign.shape[2])

    def test_unit_normalized(self, synthetic_states):
        benign, _ = synthetic_states
        atoms = _build_concept_atoms(benign, n_atoms=4)

        # Each atom should be unit-normalised.
        norms = atoms.norm(dim=-1)
        # Some may be zero-padded if n_atoms > rank, but valid ones should be ~1.0
        valid = norms > 0.5
        torch.testing.assert_close(
            norms[valid],
            torch.ones_like(norms[valid]),
            atol=1e-5, rtol=1e-5,
        )

    def test_atoms_are_orthogonal(self, synthetic_states):
        """Top PCA components should be nearly orthogonal."""
        benign, _ = synthetic_states
        atoms = _build_concept_atoms(benign, n_atoms=4)

        # Check orthogonality at one layer.
        layer_atoms = atoms[:, 3, :]  # (4, 64)
        gram = layer_atoms @ layer_atoms.T
        off_diagonal = gram - torch.eye(4)
        assert off_diagonal.abs().max() < 0.15, (
            f"Concept atoms should be nearly orthogonal, max off-diag={off_diagonal.abs().max():.4f}"
        )


class TestSpectralResidualize:
    """Tests for _spectral_residualize."""

    def test_output_shape(self, synthetic_states):
        benign, target = synthetic_states
        refusal_vec = F.normalize(target.mean(0) - benign.mean(0), p=2, dim=1)
        atoms = _build_concept_atoms(benign, n_atoms=4)

        cleaned = _spectral_residualize(refusal_vec, atoms, ridge_alpha=0.01)
        assert cleaned.shape == refusal_vec.shape

    def test_orthogonality_to_atoms(self, synthetic_states):
        """After SRA, the cleaned vector should have reduced projection onto atoms."""
        benign, target = synthetic_states
        refusal_vec = F.normalize(target.mean(0) - benign.mean(0), p=2, dim=1)
        atoms = _build_concept_atoms(benign, n_atoms=4)

        # Measure projection before cleaning.
        proj_before = 0.0
        for i in range(atoms.shape[0]):
            proj_before += (refusal_vec * atoms[i]).sum(dim=-1).abs().mean().item()

        cleaned = _spectral_residualize(refusal_vec, atoms, ridge_alpha=0.001)

        proj_after = 0.0
        for i in range(atoms.shape[0]):
            proj_after += (cleaned * atoms[i]).sum(dim=-1).abs().mean().item()

        assert proj_after < proj_before, (
            f"Projection should decrease: before={proj_before:.4f}, after={proj_after:.4f}"
        )

    def test_preserves_nontrivial_direction(self, synthetic_states):
        """The cleaned vector should not collapse to zero."""
        benign, target = synthetic_states
        refusal_vec = F.normalize(target.mean(0) - benign.mean(0), p=2, dim=1)
        atoms = _build_concept_atoms(benign, n_atoms=4)

        cleaned = _spectral_residualize(refusal_vec, atoms, ridge_alpha=0.01)
        norms = cleaned.norm(dim=-1)

        assert norms.mean() > 0.1, f"Cleaned vector should not be zero: mean norm={norms.mean():.4f}"

    def test_ridge_alpha_effect(self, synthetic_states):
        """Large alpha should preserve the original vector more."""
        benign, target = synthetic_states
        refusal_vec = F.normalize(target.mean(0) - benign.mean(0), p=2, dim=1)
        atoms = _build_concept_atoms(benign, n_atoms=4)

        cleaned_low = _spectral_residualize(refusal_vec, atoms, ridge_alpha=0.001)
        cleaned_high = _spectral_residualize(refusal_vec, atoms, ridge_alpha=100.0)

        # High alpha should produce vectors closer to the original.
        diff_low = (cleaned_low - refusal_vec).norm()
        diff_high = (cleaned_high - refusal_vec).norm()

        assert diff_high < diff_low, (
            f"High alpha should preserve more: diff_high={diff_high:.4f}, diff_low={diff_low:.4f}"
        )


class TestComputeSRAVectors:
    """Tests for compute_sra_vectors end-to-end."""

    def test_output_shape(self, synthetic_states):
        benign, target = synthetic_states
        vectors = compute_sra_vectors(benign, target, base_method=VectorMethod.MEAN)
        assert vectors.shape == (benign.shape[1], benign.shape[2])

    def test_unit_normalized(self, synthetic_states):
        benign, target = synthetic_states
        vectors = compute_sra_vectors(benign, target, base_method=VectorMethod.MEAN)
        norms = vectors.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_differs_from_base_method(self, synthetic_states):
        """SRA-cleaned vectors should differ from the raw base method vectors."""
        from abliterix.vectors import compute_steering_vectors

        benign, target = synthetic_states
        base = compute_steering_vectors(benign, target, VectorMethod.MEAN, orthogonal_projection=False)
        sra = compute_sra_vectors(benign, target, base_method=VectorMethod.MEAN)

        diff = (sra - base).abs().max().item()
        assert diff > 1e-4, f"SRA should differ from base: max diff={diff}"
