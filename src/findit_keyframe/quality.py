"""Per-frame quality scoring, optimized for batch candidate ranking.

Why the design looks the way it does
------------------------------------

A candidate frame is scored for exactly three things:

1. Sharpness (Tenengrad = mean of squared Sobel gradients)
2. Brightness (luma mean)
3. Variance (luma variance — solid-color detector)

Naive implementation computes each on the full-resolution RGB frame and
pays twice for the brightness/variance because the filter step and the
reporting step both need them. This module solves that with two moves:

* **Downsample to a fixed small square before scoring.** Quality metrics
  are scale-invariant for the purpose of *ranking candidates within a
  bucket* — the sharpest 1080p frame is still the sharpest at 384 px. A
  5× linear downscale delivers ~25× less compute for Sobel and stats.

* **Single-pass ``score_frame``.** All three metrics come out of one call,
  bound into a frozen ``FrameScore`` dataclass. The filter step consumes
  the struct and does not recompute anything.

The output [`Keyframe`](types.py) still carries the original full-resolution
PIL image — we only downscale for the *scoring* temporary, never the
artefact we hand downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

# ---- Public constants --------------------------------------------------------

#: Longest-side pixel dimension used for quality evaluation.  384 matches the
#: native input size of SigLIP 2 base, so this is also roughly the effective
#: resolution downstream sees.
QUALITY_TARGET_DIM: int = 384

#: Sobel kernel size used by Tenengrad. 3 is the OpenCV fast path.
_SOBEL_KSIZE: int = 3


LumaArray = npt.NDArray[np.uint8]
"""Single-channel 8-bit luma image, shape ``(H, W)``."""


# ---- Aggregate score ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameScore:
    """All quality metrics for a single frame, computed in a single pass.

    Attributes:
        sharpness: Tenengrad mean of squared Sobel gradients. Higher is sharper.
            Absolute scale depends on resolution of the scoring luma plane —
            this library always scores at :data:`QUALITY_TARGET_DIM`, so values
            are directly comparable across frames and shots.
        brightness: Mean luma in 0–255 space.
        variance: Luma variance. Used to flag solid-color / flat frames.
    """

    sharpness: float
    brightness: float
    variance: float

    def is_unusable(
        self,
        black_threshold: float,
        bright_threshold: float,
        variance_threshold: float,
    ) -> bool:
        """True if the frame fails any hard quality gate.

        Fails when the frame is near-black, overexposed, or nearly a solid
        color. Consumer of the score — never recomputes the metrics.
        """
        if self.brightness < black_threshold:
            return True
        if self.brightness > bright_threshold:
            return True
        return self.variance < variance_threshold


# ---- Primitive metrics (kept public for tests and advanced users) ------------


def tenengrad_sharpness(luma: LumaArray) -> float:
    """Tenengrad: mean of squared 3×3 Sobel gradient magnitudes.

    Higher value → sharper image. Absolute scale depends on resolution;
    use only for *relative* ranking within images of the same dimensions.
    """
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=_SOBEL_KSIZE)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=_SOBEL_KSIZE)
    return float(np.mean(gx * gx + gy * gy))


def luma_stats(luma: LumaArray) -> tuple[float, float]:
    """Return ``(mean, variance)`` of the luma plane in 0–255 space.

    Uses ``float64`` intermediates so numerics stay stable on 4K+ frames.
    """
    mean = float(np.mean(luma, dtype=np.float64))
    var = float(np.var(luma, dtype=np.float64))
    return mean, var


# ---- Combined pipeline -------------------------------------------------------


def downscale_for_quality(
    image: npt.NDArray[np.uint8],
    target_dim: int = QUALITY_TARGET_DIM,
) -> npt.NDArray[np.uint8]:
    """Resize so the longest side equals ``target_dim``, preserving aspect ratio.

    * Uses ``INTER_AREA`` — the correct interpolation for shrinking (averaging
      the covered source pixels gives natural anti-aliasing).
    * Returns the input unchanged when already small enough, avoiding an
      unnecessary copy.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= target_dim:
        return image
    scale = target_dim / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def score_frame(rgb: npt.NDArray[np.uint8]) -> FrameScore:
    """Compute all quality metrics for one RGB frame in a single pass.

    Pipeline:

    1. Downscale the RGB frame so the longest side is ``QUALITY_TARGET_DIM``.
       This is the dominant speedup — everything below runs on ~25× fewer
       pixels at 1080p source.
    2. Convert to luma via BT.601 (``cv2.cvtColor``).
    3. Compute Tenengrad + mean + variance from the luma plane.

    Args:
        rgb: HxWx3 uint8 RGB image.

    Returns:
        A :class:`FrameScore` with every metric the selector needs.
    """
    small_rgb = downscale_for_quality(rgb)
    luma = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    sharp = tenengrad_sharpness(luma)
    mean, var = luma_stats(luma)
    return FrameScore(sharpness=sharp, brightness=mean, variance=var)


# ---- Back-compat helpers used by tests ---------------------------------------


def rgb_to_luma(rgb: npt.NDArray[np.uint8]) -> LumaArray:
    """Convert an HxWx3 RGB uint8 image to HxW luma using BT.601."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def is_unusable_frame(
    luma: LumaArray,
    black_threshold: float,
    bright_threshold: float,
    variance_threshold: float,
) -> bool:
    """Legacy convenience: scores a luma plane and runs the hard gate.

    Prefer :func:`score_frame` + :meth:`FrameScore.is_unusable` in new code —
    this helper recomputes :func:`luma_stats` every call.
    """
    mean, var = luma_stats(luma)
    score = FrameScore(sharpness=0.0, brightness=mean, variance=var)
    return score.is_unusable(black_threshold, bright_threshold, variance_threshold)


__all__ = [
    "QUALITY_TARGET_DIM",
    "FrameScore",
    "LumaArray",
    "downscale_for_quality",
    "is_unusable_frame",
    "luma_stats",
    "rgb_to_luma",
    "score_frame",
    "tenengrad_sharpness",
]
