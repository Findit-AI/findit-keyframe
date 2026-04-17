"""Per-frame quality metrics.

All functions operate on a single-channel uint8 luma (Y) numpy array.
The caller is responsible for color-space conversion (see ``rgb_to_luma``).

Design rationale:
    * **Tenengrad** (mean of squared Sobel gradients) is chosen over Laplacian
      variance because it has better monotonicity on video frames and is
      equally cheap with OpenCV's optimized Sobel kernel.
    * Black-frame / overexposure / solid-color checks are intentionally
      cheap short-circuits so we discard unusable candidates before spending
      cycles on sharpness scoring.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

# 3x3 Sobel. Larger kernels give marginally different sensitivity; not worth
# the configurability surface at this stage.
_SOBEL_KSIZE: int = 3

LumaArray = npt.NDArray[np.uint8]
"""Single-channel 8-bit luma image, shape ``(H, W)``."""


def rgb_to_luma(rgb: npt.NDArray[np.uint8]) -> LumaArray:
    """Convert an HxWx3 RGB uint8 image to HxW luma using BT.601.

    Args:
        rgb: Image in ``(H, W, 3)`` RGB uint8.

    Returns:
        ``(H, W)`` uint8 luma array.
    """
    # BT.601: Y = 0.299 R + 0.587 G + 0.114 B. OpenCV's COLOR_RGB2GRAY uses
    # the same coefficients in its fast integer path.
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def tenengrad_sharpness(luma: LumaArray) -> float:
    """Mean of squared 3x3 Sobel gradient magnitudes.

    Higher value → sharper image. Absolute scale depends on resolution;
    use for *relative* ranking within a single shot / bucket.

    Typical ranges at 1080p:
        * < 100: visibly blurry / out-of-focus
        * 100–500: adequate for VLM
        * > 500: sharp
    """
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=_SOBEL_KSIZE)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=_SOBEL_KSIZE)
    # gx * gx + gy * gy stays non-negative so mean is well-defined.
    return float(np.mean(gx * gx + gy * gy))


def luma_stats(luma: LumaArray) -> tuple[float, float]:
    """Return ``(mean, variance)`` of the luma plane in 0–255 space."""
    # float64 intermediate keeps accuracy on 4K frames (~8M pixels).
    mean = float(np.mean(luma, dtype=np.float64))
    var = float(np.var(luma, dtype=np.float64))
    return mean, var


def is_unusable_frame(
    luma: LumaArray,
    black_threshold: float,
    bright_threshold: float,
    variance_threshold: float,
) -> bool:
    """Fast reject: frame is too dark, too bright, or effectively solid color.

    Any of these conditions mark the frame unusable for VLM consumption:

    * Mean luma below ``black_threshold`` → near-black (typical: 15).
    * Mean luma above ``bright_threshold`` → overexposed / faded-to-white
      (typical: 240).
    * Variance below ``variance_threshold`` → solid color / flat test pattern
      (typical: 5).
    """
    mean, var = luma_stats(luma)
    if mean < black_threshold:
        return True
    if mean > bright_threshold:
        return True
    return var < variance_threshold


__all__ = [
    "LumaArray",
    "rgb_to_luma",
    "tenengrad_sharpness",
    "luma_stats",
    "is_unusable_frame",
]
