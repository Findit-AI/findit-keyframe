"""Tests for quality metrics.

We use small synthetic images rather than real video: quality is a
pure-function concept, and real-video tests live in test_integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from findit_keyframe.quality import (
    FrameScore,
    QUALITY_TARGET_DIM,
    downscale_for_quality,
    is_unusable_frame,
    luma_stats,
    rgb_to_luma,
    score_frame,
    tenengrad_sharpness,
)


# ---------- tenengrad_sharpness ------------------------------------------------


def test_flat_image_has_zero_sharpness() -> None:
    """A solid-color image has no gradients → score = 0."""
    img = np.full((64, 64), 128, dtype=np.uint8)
    assert tenengrad_sharpness(img) == pytest.approx(0.0, abs=1e-6)


def test_sharp_edge_has_high_score() -> None:
    """A half-black / half-white image has strong gradients."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = 255
    # On a 64x64 half-step image the Tenengrad score is in the thousands.
    assert tenengrad_sharpness(img) > 1000.0


def test_structured_image_beats_its_blurred_version() -> None:
    """A structured image should outscore the same image after Gaussian blur.

    Note: Tenengrad measures gradient *density*, not "edge presence". Random
    uniform noise can actually outscore a single hard edge because every
    pixel contributes gradient energy. That's correct behaviour — the metric
    is specifically designed to rank an image against blurred versions of
    itself, which is exactly what we use it for (ranking frames within one
    bucket).
    """
    import cv2

    # A checkerboard has lots of structure.
    board = np.zeros((128, 128), dtype=np.uint8)
    board[::16, :] = 255
    board[:, ::16] = 255

    sharp = tenengrad_sharpness(board)
    # Heavy blur destroys the edges.
    blurred = tenengrad_sharpness(cv2.GaussianBlur(board, (15, 15), sigmaX=8.0))
    assert sharp > blurred * 2


def test_gaussian_blur_reduces_score() -> None:
    """Blurring a sharp edge must strictly reduce Tenengrad."""
    import cv2

    edge = np.zeros((128, 128), dtype=np.uint8)
    edge[:, 64:] = 255

    sharp_score = tenengrad_sharpness(edge)
    blurred = cv2.GaussianBlur(edge, (15, 15), sigmaX=5.0)
    blurred_score = tenengrad_sharpness(blurred)

    assert blurred_score < sharp_score
    # Blur should make it dramatically lower (not just marginally).
    assert blurred_score < sharp_score * 0.5


# ---------- luma_stats ---------------------------------------------------------


def test_luma_stats_constant_image() -> None:
    img = np.full((32, 32), 200, dtype=np.uint8)
    mean, var = luma_stats(img)
    assert mean == pytest.approx(200.0)
    assert var == pytest.approx(0.0, abs=1e-6)


def test_luma_stats_random_image() -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(50, 200, (100, 100), dtype=np.uint8)
    mean, var = luma_stats(img)
    # Sanity check the range (within tolerance for sample size).
    assert 100.0 < mean < 150.0
    assert var > 0.0


# ---------- is_unusable_frame --------------------------------------------------


def test_black_frame_is_unusable() -> None:
    img = np.zeros((32, 32), dtype=np.uint8)
    assert is_unusable_frame(img, black_threshold=15.0, bright_threshold=240.0, variance_threshold=5.0)


def test_overexposed_frame_is_unusable() -> None:
    img = np.full((32, 32), 250, dtype=np.uint8)
    assert is_unusable_frame(img, black_threshold=15.0, bright_threshold=240.0, variance_threshold=5.0)


def test_solid_color_frame_is_unusable() -> None:
    # Gray frame with essentially zero variance.
    img = np.full((32, 32), 128, dtype=np.uint8)
    assert is_unusable_frame(img, black_threshold=15.0, bright_threshold=240.0, variance_threshold=5.0)


def test_normal_frame_is_usable() -> None:
    rng = np.random.default_rng(42)
    img = rng.integers(50, 200, (32, 32), dtype=np.uint8)
    assert not is_unusable_frame(
        img, black_threshold=15.0, bright_threshold=240.0, variance_threshold=5.0
    )


# ---------- rgb_to_luma --------------------------------------------------------


def test_rgb_to_luma_gray_input() -> None:
    """Gray RGB = (128, 128, 128) → luma ≈ 128."""
    rgb = np.full((16, 16, 3), 128, dtype=np.uint8)
    luma = rgb_to_luma(rgb)
    assert luma.shape == (16, 16)
    assert luma.dtype == np.uint8
    # BT.601 weights sum to 1.0, so gray RGB → identical luma.
    assert np.all(luma == 128)


def test_rgb_to_luma_pure_green_weight() -> None:
    """Pure green should dominate luma (0.587 in BT.601)."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 1] = 255  # G channel
    luma = rgb_to_luma(rgb)
    # 0.587 * 255 ≈ 149. OpenCV's integer path lands on 150.
    assert 145 <= luma.mean() <= 155


# ---------- score_frame -------------------------------------------------------


def test_score_frame_on_sharp_image() -> None:
    """A checker-pattern RGB image scores above the default min_sharpness (100)."""
    rgb = np.zeros((240, 240, 3), dtype=np.uint8)
    rgb[::16, :, :] = 255
    rgb[:, ::16, :] = 255
    score = score_frame(rgb)
    assert score.sharpness > 100.0
    # Sanity checks on auxiliary fields.
    assert 0.0 <= score.brightness <= 255.0
    assert score.variance > 0.0


def test_score_frame_on_black_image() -> None:
    rgb = np.zeros((240, 240, 3), dtype=np.uint8)
    score = score_frame(rgb)
    assert score.sharpness == pytest.approx(0.0, abs=1e-6)
    assert score.brightness == pytest.approx(0.0, abs=1e-6)
    # FrameScore.is_unusable should catch this.
    assert score.is_unusable(
        black_threshold=15.0, bright_threshold=240.0, variance_threshold=5.0
    )


def test_score_frame_on_flat_gray_image() -> None:
    rgb = np.full((240, 240, 3), 128, dtype=np.uint8)
    score = score_frame(rgb)
    assert score.brightness == pytest.approx(128.0, abs=1.0)
    assert score.variance == pytest.approx(0.0, abs=1e-6)
    assert score.is_unusable(
        black_threshold=15.0, bright_threshold=240.0, variance_threshold=5.0
    )


# ---------- downscale_for_quality ---------------------------------------------


def test_downscale_small_input_unchanged() -> None:
    """Images already at or below target_dim are returned as-is (no copy)."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = downscale_for_quality(img, target_dim=QUALITY_TARGET_DIM)
    # Longest side 200 < 384 → no-op; same object returned.
    assert out is img


def test_downscale_large_input_preserves_aspect_ratio() -> None:
    # 1080x1920, longest side 1920 → scales to 384 on long side.
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = downscale_for_quality(img, target_dim=384)
    h, w = out.shape[:2]
    assert max(h, w) == 384
    # Aspect ratio roughly preserved.
    ratio_before = 1920 / 1080
    ratio_after = w / h
    assert abs(ratio_after - ratio_before) < 0.02
