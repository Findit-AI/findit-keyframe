"""Per-frame quality scoring, optimized for batch candidate ranking.

Why the design looks the way it does
------------------------------------

Each candidate frame is scored for five things:

* **Sharpness** — Tenengrad (mean of squared 3×3 Sobel gradients) on luma.
* **Brightness** — mean luma.
* **Luma variance** — for base "solid color" detection.
* **Saturation variance** — catches equiluminant multi-color frames that the
  luma variance alone would mislabel as flat.
* **Clipping ratio** — fraction of pixels where ``max(R, G, B)`` is below 5
  or above 250. Catches blown highlights and crushed shadows that the luma
  mean alone misses (e.g. a frame with a blown-out sky occupying 40 % of
  the area but a moderate overall luma mean).

Design pillars:

1. **Single pass.** All five metrics come out of one call, bound into a
   frozen :class:`FrameScore`. No downstream consumer re-computes anything.
2. **Downscale first.** Quality metrics are evaluated on a
   :data:`QUALITY_TARGET_DIM`-px longest-side shrink. Scale-invariant for
   *ranking* within a bucket. Dominant speedup: ~25× fewer pixels at 1080p.
3. **Hot operations on the right representation.** Sharpness on luma
   (HVS-aligned), clipping on RGB (caught by the channel max, not the
   weighted luma), saturation variance on HSV (the one real edge case
   luma variance misses).
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

#: Pixel value threshold below which a pixel is considered "crushed shadow".
_CLIP_LOW: int = 5
#: Pixel value threshold above which a pixel is considered "blown highlight".
_CLIP_HIGH: int = 250

#: Sobel kernel size used by Tenengrad. 3 is the OpenCV fast path.
_SOBEL_KSIZE: int = 3


LumaArray = npt.NDArray[np.uint8]
"""Single-channel 8-bit luma image, shape ``(H, W)``."""


# ---- Aggregate score ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameScore:
    """All quality metrics for a single frame, computed in a single pass.

    Attributes:
        sharpness: Tenengrad (mean of squared 3×3 Sobel gradients) on luma.
            Scored at :data:`QUALITY_TARGET_DIM` so values are directly
            comparable across frames.
        brightness: Mean luma in 0–255 space.
        luma_variance: Variance of the luma plane.
        sat_variance: Variance of the HSV saturation plane. Used in
            conjunction with :attr:`luma_variance` to detect truly-flat
            frames — only when **both** are low is the frame considered flat
            (avoids flagging equiluminant multi-color frames).
        clipping: Fraction of pixels where ``max(R, G, B)`` is clipped
            (below :data:`_CLIP_LOW` or above :data:`_CLIP_HIGH`). Range
            ``[0.0, 1.0]``.
    """

    sharpness: float
    brightness: float
    luma_variance: float
    sat_variance: float
    clipping: float

    def is_unusable(
        self,
        black_threshold: float,
        bright_threshold: float,
        luma_variance_threshold: float,
        sat_variance_threshold: float,
        max_clipping: float,
    ) -> bool:
        """True if the frame should be rejected before ranking.

        Hard gates (any one trips → unusable):

        * Mean luma below ``black_threshold`` (near-black).
        * Mean luma above ``bright_threshold`` (overexposed).
        * **Both** luma and saturation variance below thresholds (truly flat —
          not just equiluminant). This AND is intentional: a frame of solid
          colors at the same luma level has ``luma_variance ≈ 0`` but
          ``sat_variance > 0`` and should be kept.
        * Clipping ratio above ``max_clipping`` (blown highlights / crushed
          shadows on a large fraction of the frame).
        """
        if self.brightness < black_threshold:
            return True
        if self.brightness > bright_threshold:
            return True
        if (
            self.luma_variance < luma_variance_threshold
            and self.sat_variance < sat_variance_threshold
        ):
            return True
        return self.clipping > max_clipping


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
    """Return ``(mean, variance)`` of the luma plane in 0–255 space."""
    mean = float(np.mean(luma, dtype=np.float64))
    var = float(np.var(luma, dtype=np.float64))
    return mean, var


def clipping_ratio(rgb: npt.NDArray[np.uint8]) -> float:
    """Fraction of pixels where ``max(R, G, B)`` is clipped low or high.

    Catches the two flavors of blown exposure the luma mean alone misses:

    * Blown highlights: ``max(R, G, B) > 250`` — often a saturated colour
      rather than pure white, so the luma mean may still be moderate.
    * Crushed shadows: ``max(R, G, B) < 5`` — consistent "no signal in any
      channel" regions.

    Args:
        rgb: HxWx3 uint8 RGB or BGR image (the max-per-channel is
            byte-order-agnostic).

    Returns:
        Fraction in ``[0.0, 1.0]``. Note this is an *inclusive* count — a
        single clipped pixel contributes proportionally.
    """
    max_channel = rgb.max(axis=2)
    over = max_channel > _CLIP_HIGH
    under = max_channel < _CLIP_LOW
    return float((over | under).mean())


# ---- Combined pipeline -------------------------------------------------------


def downscale_for_quality(
    image: npt.NDArray[np.uint8],
    target_dim: int = QUALITY_TARGET_DIM,
) -> npt.NDArray[np.uint8]:
    """Resize so the longest side equals ``target_dim``, preserving aspect ratio.

    * Uses ``INTER_AREA`` — the correct interpolation for shrinking.
    * Returns the input unchanged when already small enough.
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
    """Compute all five quality metrics for one RGB frame in a single pass.

    Pipeline:

    1. Downscale the RGB frame so the longest side is ``QUALITY_TARGET_DIM``.
       Every subsequent op runs on this cheaper copy.
    2. Convert to luma (BT.601) and HSV via OpenCV.
    3. Compute Tenengrad + luma mean/variance + saturation variance +
       clipping ratio.

    Args:
        rgb: HxWx3 uint8 RGB image.
    """
    small_rgb = downscale_for_quality(rgb)
    luma = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)

    sharp = tenengrad_sharpness(luma)
    mean, luma_var = luma_stats(luma)
    sat_var = float(np.var(hsv[:, :, 1], dtype=np.float64))
    clip = clipping_ratio(small_rgb)

    return FrameScore(
        sharpness=sharp,
        brightness=mean,
        luma_variance=luma_var,
        sat_variance=sat_var,
        clipping=clip,
    )


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
    """Legacy luma-only unusable check.

    This helper predates the clipping and saturation-variance signals — it
    only knows about luma mean and luma variance. New code should prefer
    :func:`score_frame` + :meth:`FrameScore.is_unusable`.
    """
    mean, var = luma_stats(luma)
    if mean < black_threshold:
        return True
    if mean > bright_threshold:
        return True
    return var < variance_threshold


__all__ = [
    "QUALITY_TARGET_DIM",
    "FrameScore",
    "LumaArray",
    "clipping_ratio",
    "downscale_for_quality",
    "is_unusable_frame",
    "luma_stats",
    "rgb_to_luma",
    "score_frame",
    "tenengrad_sharpness",
]
