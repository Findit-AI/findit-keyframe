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


# Default thresholds mirroring Config defaults — kept local so the tests
# don't depend on Config evolving.
_BLACK_TH = 15.0
_BRIGHT_TH = 240.0
_LUMA_VAR_TH = 5.0
_SAT_VAR_TH = 3.0
_MAX_CLIPPING = 0.50


def _is_unusable(score) -> bool:  # type: ignore[no-untyped-def]
    return score.is_unusable(
        _BLACK_TH, _BRIGHT_TH, _LUMA_VAR_TH, _SAT_VAR_TH, _MAX_CLIPPING
    )


def test_score_frame_on_sharp_image() -> None:
    """A checker-pattern RGB image scores above the default min_sharpness (100)."""
    rgb = np.zeros((240, 240, 3), dtype=np.uint8)
    rgb[::16, :, :] = 255
    rgb[:, ::16, :] = 255
    score = score_frame(rgb)
    assert score.sharpness > 100.0
    # Sanity checks on auxiliary fields.
    assert 0.0 <= score.brightness <= 255.0
    assert score.luma_variance > 0.0
    assert 0.0 <= score.clipping <= 1.0


def test_score_frame_on_black_image() -> None:
    rgb = np.zeros((240, 240, 3), dtype=np.uint8)
    score = score_frame(rgb)
    assert score.sharpness == pytest.approx(0.0, abs=1e-6)
    assert score.brightness == pytest.approx(0.0, abs=1e-6)
    # Clipping should also hit 1.0 (all pixels are max<5).
    assert score.clipping == pytest.approx(1.0, abs=1e-6)
    # Both the brightness gate and the clipping gate catch this.
    assert _is_unusable(score)


def test_score_frame_on_flat_gray_image() -> None:
    rgb = np.full((240, 240, 3), 128, dtype=np.uint8)
    score = score_frame(rgb)
    assert score.brightness == pytest.approx(128.0, abs=1.0)
    assert score.luma_variance == pytest.approx(0.0, abs=1e-6)
    assert score.sat_variance == pytest.approx(0.0, abs=1e-6)
    # Both luma AND saturation variance are zero → truly flat.
    assert _is_unusable(score)


def test_equiluminant_multi_color_is_NOT_flagged_flat() -> None:
    """A frame with two equiluminant patches of different color must survive.

    Left half saturated red ``(255, 0, 0)``: Y≈76, S=255.
    Right half gray ``(76, 76, 76)``: Y=76, S=0.

    Luma variance is effectively zero (both halves at Y=76), but saturation
    varies dramatically (255 vs 0). The AND'd flat check should keep this
    frame — a luma-only detector would wrongly reject it.
    """
    rgb = np.zeros((240, 240, 3), dtype=np.uint8)
    rgb[:, :120] = (255, 0, 0)        # saturated red, Y≈76, S=255
    rgb[:, 120:] = (76, 76, 76)        # gray, Y=76, S=0
    score = score_frame(rgb)
    # Luma variance should be tiny — this is the equiluminant trap.
    assert score.luma_variance < 20.0
    # Saturation variance should be large (fully saturated vs unsaturated).
    assert score.sat_variance > 1000.0
    # Because AND'd together, the frame must NOT be marked unusable.
    assert not _is_unusable(score)


def test_blown_highlight_is_flagged_via_clipping() -> None:
    """Frame with a large blown-out region triggers the clipping gate.

    The overall luma mean sits inside ``[black, bright]``, so the legacy
    mean-only check would NOT reject this. Clipping does.
    """
    # 240x240 frame with 60% of area blown to pure white — past the
    # default max_clipping=0.50 gate.
    rgb = np.full((240, 240, 3), 90, dtype=np.uint8)
    rgb[:, 96:, :] = 255  # right 60% = 144/240 columns blown white
    score = score_frame(rgb)
    # Mean luma sits inside the [15, 240] legal range.
    assert 15 < score.brightness < 240
    # Clipping exceeds the default gate.
    assert score.clipping > 0.50
    assert _is_unusable(score)


def test_high_contrast_scene_survives_clipping_gate() -> None:
    """A normal high-contrast scene (small bright area) should not be rejected."""
    rgb = np.full((240, 240, 3), 120, dtype=np.uint8)
    # Only 10% of frame is "bright sky".
    rgb[:24, :, :] = 252
    score = score_frame(rgb)
    # < 50% clipped at default.
    assert score.clipping < 0.20
    assert not _is_unusable(score)


def test_clipping_ratio_is_symmetric() -> None:
    """Both blown highlights and crushed shadows count as clipping."""
    from findit_keyframe.quality import clipping_ratio

    mid = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert clipping_ratio(mid) == pytest.approx(0.0, abs=1e-6)

    blown = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert clipping_ratio(blown) == pytest.approx(1.0, abs=1e-6)

    crushed = np.full((100, 100, 3), 0, dtype=np.uint8)
    assert clipping_ratio(crushed) == pytest.approx(1.0, abs=1e-6)

    # Half and half should be ~0.5 regardless of direction.
    mixed = np.full((100, 100, 3), 128, dtype=np.uint8)
    mixed[:, :50] = 0
    assert clipping_ratio(mixed) == pytest.approx(0.5, abs=1e-3)


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
