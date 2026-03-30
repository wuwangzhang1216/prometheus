"""Tests for Spherical Steering (geodesic rotation on the activation hypersphere)."""

import math

import torch
import torch.nn.functional as F

from abliterix.core.steering import _make_spherical_hook, _make_angular_hook


class TestSphericalHook:
    """Tests for _make_spherical_hook."""

    def test_preserves_norm(self):
        """Spherical steering must preserve the activation norm."""
        torch.manual_seed(42)
        d = F.normalize(torch.randn(64), p=2, dim=0)
        hook = _make_spherical_hook(d, angle_degrees=90.0)

        h = torch.randn(2, 10, 64)  # (batch, seq, hidden)
        original_norms = h.norm(dim=-1)

        # Simulate hook call.
        h_new = hook(None, None, h)

        new_norms = h_new.norm(dim=-1)
        torch.testing.assert_close(new_norms, original_norms, atol=1e-4, rtol=1e-4)

    def test_moves_toward_direction(self):
        """After steering, activations should have larger projection onto d."""
        torch.manual_seed(42)
        d = F.normalize(torch.randn(64), p=2, dim=0)
        hook = _make_spherical_hook(d, angle_degrees=45.0)

        h = torch.randn(1, 5, 64)
        proj_before = (F.normalize(h, p=2, dim=-1) @ d).mean()

        h_new = hook(None, None, h)
        proj_after = (F.normalize(h_new, p=2, dim=-1) @ d).mean()

        assert proj_after > proj_before, (
            f"Projection should increase: before={proj_before:.4f}, after={proj_after:.4f}"
        )

    def test_zero_angle_is_identity(self):
        """With angle=0, spherical steering should return the original activation."""
        torch.manual_seed(42)
        d = F.normalize(torch.randn(64), p=2, dim=0)
        hook = _make_spherical_hook(d, angle_degrees=0.0)

        h = torch.randn(1, 3, 64)
        h_new = hook(None, None, h)

        torch.testing.assert_close(h_new, h, atol=1e-5, rtol=1e-5)

    def test_equivalent_to_angular_for_2d_rotation(self):
        """Spherical and angular produce equivalent results for 2D rotation.

        Both methods rotate in the plane spanned by h and d, which is
        mathematically equivalent to geodesic rotation on the hypersphere.
        This test validates that our spherical implementation is correct by
        checking consistency with the known-good angular implementation.
        """
        torch.manual_seed(42)
        d = F.normalize(torch.randn(64), p=2, dim=0)

        spherical_hook = _make_spherical_hook(d, angle_degrees=45.0)
        angular_hook = _make_angular_hook(d, angle_degrees=45.0, adaptive=False)

        h = torch.randn(1, 10, 64)
        h_spherical = spherical_hook(None, None, h.clone())
        h_angular = angular_hook(None, None, h.clone())

        # The two methods should produce (nearly) identical results.
        torch.testing.assert_close(h_spherical, h_angular, atol=1e-4, rtol=1e-4)

    def test_large_angle_inverts_direction(self):
        """A 180-degree rotation should invert the projection onto d."""
        torch.manual_seed(42)
        d = F.normalize(torch.randn(64), p=2, dim=0)
        hook = _make_spherical_hook(d, angle_degrees=180.0)

        h = torch.randn(1, 5, 64)
        proj_before = (h @ d).mean()

        h_new = hook(None, None, h)
        proj_after = (h_new @ d).mean()

        # Projection should flip sign.
        assert proj_before * proj_after < 0, (
            f"180° should invert: before={proj_before:.4f}, after={proj_after:.4f}"
        )

    def test_handles_tuple_output(self):
        """Hook should handle tuple outputs (as returned by some transformer layers)."""
        torch.manual_seed(42)
        d = F.normalize(torch.randn(64), p=2, dim=0)
        hook = _make_spherical_hook(d, angle_degrees=45.0)

        h = torch.randn(1, 3, 64)
        extra = torch.randn(1, 3, 64)  # e.g., attention weights
        output = (h, extra)

        result = hook(None, None, output)
        assert isinstance(result, tuple)
        assert len(result) == 2
        # The second element should be unchanged.
        torch.testing.assert_close(result[1], extra)

    def test_numerical_stability_near_parallel(self):
        """Hook should handle activations nearly parallel to direction."""
        d = F.normalize(torch.randn(64), p=2, dim=0)
        hook = _make_spherical_hook(d, angle_degrees=30.0)

        # Make h very close to d.
        h = d.unsqueeze(0).unsqueeze(0) * 2.5 + torch.randn(1, 1, 64) * 1e-6
        h_new = hook(None, None, h)

        # Should not produce NaN or Inf.
        assert not torch.isnan(h_new).any(), "NaN detected in spherical steering output"
        assert not torch.isinf(h_new).any(), "Inf detected in spherical steering output"
