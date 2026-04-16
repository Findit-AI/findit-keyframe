"""Tests for ``findit_keyframe.quality``.

Every numeric expectation is computed by hand from the spec in
``TASKS.md`` §4 and ``docs/algorithm.md`` so the Rust port can replay the
same assertions against the same inputs.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from findit_keyframe.quality import (
    QualityGate,
    compute_quality,
    entropy,
    laplacian_variance,
    luma_variance,
    mean_luma,
    rgb_to_luma,
)
from findit_keyframe.types import QualityMetrics


def _solid_rgb(value: int, h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# rgb_to_luma                                                                 #
# --------------------------------------------------------------------------- #


class TestRgbToLuma:
    def test_rejects_non_uint8(self):
        with pytest.raises(ValueError, match="uint8"):
            rgb_to_luma(np.zeros((4, 4, 3), dtype=np.float32))

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="shape"):
            rgb_to_luma(np.zeros((4, 4), dtype=np.uint8))

    def test_rejects_4_channel(self):
        with pytest.raises(ValueError, match="shape"):
            rgb_to_luma(np.zeros((4, 4, 4), dtype=np.uint8))

    def test_pure_black_yields_16(self):
        # BT.601 limited range: black -> Y = ((0+0+0+128) >> 8) + 16 = 16.
        y = rgb_to_luma(_solid_rgb(0))
        assert y.dtype == np.uint8
        assert y.shape == (32, 32)
        assert (y == 16).all()

    def test_pure_white_yields_235(self):
        # Y = ((66+129+25)*255 + 128) >> 8 + 16 = 56228 >> 8 + 16 = 219 + 16 = 235.
        y = rgb_to_luma(_solid_rgb(255))
        assert (y == 235).all()

    def test_pure_red(self):
        # Y = ((66*255 + 128) >> 8) + 16 = 16958 >> 8 + 16 = 66 + 16 = 82.
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        assert (rgb_to_luma(rgb) == 82).all()

    def test_pure_green(self):
        # Y = ((129*255 + 128) >> 8) + 16 = 33023 >> 8 + 16 = 128 + 16 = 144.
        # (33023 = 128 * 256 + 255, so the shift truncates to 128.)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 1] = 255
        assert (rgb_to_luma(rgb) == 144).all()

    def test_pure_blue(self):
        # Y = ((25*255 + 128) >> 8) + 16 = 6503 >> 8 + 16 = 25 + 16 = 41.
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[..., 2] = 255
        assert (rgb_to_luma(rgb) == 41).all()


# --------------------------------------------------------------------------- #
# laplacian_variance                                                          #
# --------------------------------------------------------------------------- #


class TestLaplacianVariance:
    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2D"):
            laplacian_variance(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_rejects_too_small(self):
        with pytest.raises(ValueError, match="3x3"):
            laplacian_variance(np.zeros((2, 2), dtype=np.uint8))

    def test_uniform_image_zero(self):
        luma = np.full((16, 16), 128, dtype=np.uint8)
        assert laplacian_variance(luma) == 0.0

    def test_random_noise_is_high(self):
        rng = np.random.default_rng(seed=42)
        luma = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        assert laplacian_variance(luma) > 1000.0

    def test_smooth_gradient_is_low(self):
        # Linear horizontal ramp: the discrete second derivative is exactly 0
        # in the interior. Laplacian variance therefore collapses to ~0.
        luma = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        assert laplacian_variance(luma) < 1.0

    def test_isolated_spike_known(self):
        # Single bright pixel at the centre of a 5x5 zero field.
        # Filtered values across the 3x3 interior:
        #   corners = 0, edges = +100, centre = -400
        # mean = 0, variance (ddof=0) = (4*10000 + 160000) / 9 = 200000/9.
        luma = np.zeros((5, 5), dtype=np.uint8)
        luma[2, 2] = 100
        assert laplacian_variance(luma) == pytest.approx(200000.0 / 9.0)

    def test_returns_finite(self):
        rng = np.random.default_rng(seed=0)
        luma = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
        assert math.isfinite(laplacian_variance(luma))


# --------------------------------------------------------------------------- #
# mean_luma                                                                   #
# --------------------------------------------------------------------------- #


class TestMeanLuma:
    def test_uniform_zero(self):
        assert mean_luma(np.zeros((4, 4), dtype=np.uint8)) == 0.0

    def test_uniform_255(self):
        assert mean_luma(np.full((4, 4), 255, dtype=np.uint8)) == 1.0

    def test_uniform_128(self):
        assert mean_luma(np.full((4, 4), 128, dtype=np.uint8)) == pytest.approx(128 / 255)


# --------------------------------------------------------------------------- #
# luma_variance                                                               #
# --------------------------------------------------------------------------- #


class TestLumaVariance:
    def test_uniform_zero(self):
        assert luma_variance(np.full((8, 8), 128, dtype=np.uint8)) == 0.0

    def test_two_value_known(self):
        # Sample variance (ddof=1) of [0, 255, 0, 255]:
        #   mean = 127.5
        #   sum((x - mean)^2) = 4 * 127.5^2 = 65025
        #   variance = 65025 / (4 - 1) = 21675
        luma = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        assert luma_variance(luma) == pytest.approx(21675.0)


# --------------------------------------------------------------------------- #
# entropy                                                                     #
# --------------------------------------------------------------------------- #


class TestEntropy:
    def test_uniform_distribution_max(self):
        # Each value 0..255 appears once -> H = log2(256) = 8.
        luma = np.arange(256, dtype=np.uint8).reshape(16, 16)
        assert entropy(luma) == pytest.approx(8.0)

    def test_constant_zero(self):
        # Delta distribution -> H = 0.
        luma = np.full((16, 16), 100, dtype=np.uint8)
        assert entropy(luma) == 0.0

    def test_two_value_one_bit(self):
        # Half 0, half 255 with equal counts -> H = 1.
        luma = np.array([[0] * 8, [255] * 8] * 8, dtype=np.uint8)
        assert entropy(luma) == pytest.approx(1.0)

    def test_custom_bins(self):
        # 4 bins, uniform across them -> H = log2(4) = 2.
        luma = np.array([0, 64, 128, 192] * 4, dtype=np.uint8).reshape(4, 4)
        assert entropy(luma, bins=4) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# QualityGate                                                                 #
# --------------------------------------------------------------------------- #


class TestQualityGate:
    def _metrics(self, **overrides):
        defaults = {
            "laplacian_var": 100.0,
            "mean_luma": 0.5,
            "luma_variance": 1000.0,
            "entropy": 7.0,
            "saliency_mass": 0.0,
        }
        return QualityMetrics(**(defaults | overrides))

    def test_default_thresholds_match_spec(self):
        gate = QualityGate()
        assert gate.min_mean_luma == pytest.approx(15.0 / 255.0)
        assert gate.max_mean_luma == pytest.approx(240.0 / 255.0)
        assert gate.min_luma_variance == 5.0

    def test_normal_frame_passes(self):
        assert QualityGate().passes(self._metrics()) is True

    def test_too_dark_rejected(self):
        assert QualityGate().passes(self._metrics(mean_luma=0.05)) is False

    def test_too_bright_rejected(self):
        assert QualityGate().passes(self._metrics(mean_luma=0.99)) is False

    def test_flat_rejected(self):
        assert QualityGate().passes(self._metrics(luma_variance=4.99)) is False

    def test_lower_mean_boundary_inclusive(self):
        assert QualityGate().passes(self._metrics(mean_luma=15.0 / 255.0)) is True

    def test_upper_mean_boundary_inclusive(self):
        assert QualityGate().passes(self._metrics(mean_luma=240.0 / 255.0)) is True

    def test_variance_boundary_inclusive(self):
        assert QualityGate().passes(self._metrics(luma_variance=5.0)) is True


# --------------------------------------------------------------------------- #
# compute_quality                                                             #
# --------------------------------------------------------------------------- #


class TestComputeQuality:
    def test_returns_quality_metrics(self):
        rgb = np.full((16, 16, 3), 128, dtype=np.uint8)
        assert isinstance(compute_quality(rgb), QualityMetrics)

    def test_default_saliency_is_zero(self):
        rgb = np.full((16, 16, 3), 128, dtype=np.uint8)
        assert compute_quality(rgb).saliency_mass == 0.0

    def test_saliency_passed_through(self):
        rgb = np.full((16, 16, 3), 128, dtype=np.uint8)
        assert compute_quality(rgb, saliency=0.42).saliency_mass == pytest.approx(0.42)

    def test_black_frame_metrics(self):
        # All-black RGB -> Y = 16 everywhere; everything spreading to zero.
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        m = compute_quality(rgb)
        assert m.mean_luma == pytest.approx(16 / 255)
        assert m.luma_variance == 0.0
        assert m.laplacian_var == 0.0
        assert m.entropy == 0.0
        # Gate rejects on luma_variance < 5.
        assert QualityGate().passes(m) is False

    def test_random_noise_passes_gate(self):
        rng = np.random.default_rng(seed=0)
        rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        m = compute_quality(rgb)
        assert m.luma_variance > 5.0
        assert m.laplacian_var > 100.0
        assert QualityGate().passes(m) is True

    def test_smooth_gradient_passes_gate(self):
        ramp = np.linspace(0, 255, 64, dtype=np.uint8)
        rgb = np.stack([np.tile(ramp, (64, 1))] * 3, axis=-1)
        m = compute_quality(rgb)
        assert m.luma_variance > 100.0
        assert m.laplacian_var < 5.0
        assert QualityGate().passes(m) is True


# --------------------------------------------------------------------------- #
# Performance                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_compute_quality_performance_budget():
    """TASKS.md T4: <5 ms on M-series Mac. CI runners can be ~3x slower."""
    rgb = np.random.default_rng(0).integers(0, 256, size=(384, 384, 3), dtype=np.uint8)
    for _ in range(3):
        compute_quality(rgb)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        compute_quality(rgb)
    avg_ms = (time.perf_counter() - t0) / n * 1000
    assert avg_ms < 15.0, f"compute_quality avg {avg_ms:.2f} ms exceeds 15 ms budget"
