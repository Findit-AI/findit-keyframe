"""Per-frame quality metrics. Pure numpy, no OpenCV / scipy / skimage.

Every function is shaped for a 1:1 Rust port: scalar arithmetic on slices,
no broadcasting magic, no in-place mutation across function boundaries.
See ``docs/rust-porting.md`` §3 for the numpy-to-ndarray idiom map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from findit_keyframe.types import QualityMetrics

__all__ = [
    "QualityGate",
    "compute_quality",
    "entropy",
    "laplacian_variance",
    "luma_variance",
    "mean_luma",
    "rgb_to_luma",
]

LumaArray = npt.NDArray[np.uint8]
RgbArray = npt.NDArray[np.uint8]


def rgb_to_luma(rgb: RgbArray) -> LumaArray:
    """BT.601 limited-range fixed-point luma (``Y in [16, 235]``).

    Formula: ``((66*R + 129*G + 25*B + 128) >> 8) + 16``. Identical to
    scenesdetect's choice so a Y-plane round-trip is bit-exact across crates.
    """
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {rgb.dtype}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
    r = rgb[..., 0].astype(np.uint32)
    g = rgb[..., 1].astype(np.uint32)
    b = rgb[..., 2].astype(np.uint32)
    y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
    return y.astype(np.uint8)


def laplacian_variance(luma: LumaArray) -> float:
    """Variance of a 3x3 Laplacian-filtered luma image. Sharpness proxy.

    Kernel ``[[0, 1, 0], [1, -4, 1], [0, 1, 0]]``. One-pixel border is dropped.
    Variance uses the population denominator (N) to match OpenCV's
    ``meanStdDev(Laplacian(...))``.
    """
    if luma.ndim != 2:
        raise ValueError(f"luma must be 2D, got shape {luma.shape}")
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        raise ValueError(f"luma too small for 3x3 Laplacian: {luma.shape}")
    f = luma.astype(np.int32)
    out = f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:] - 4 * f[1:-1, 1:-1]
    return float(out.var())


def mean_luma(luma: LumaArray) -> float:
    """Mean luma normalised to ``[0.0, 1.0]``."""
    return float(luma.mean()) / 255.0


def luma_variance(luma: LumaArray) -> float:
    """Sample variance (ddof=1) of luma values on the raw 0-255 scale."""
    return float(luma.var(ddof=1))


def entropy(luma: LumaArray, bins: int = 256) -> float:
    """Shannon entropy in bits of a ``bins``-bin luma histogram over ``[0, 256)``."""
    counts, _ = np.histogram(luma, bins=bins, range=(0, 256))
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts.astype(np.float64) / float(total)
    p_nz = p[p > 0]
    return float(-np.sum(p_nz * np.log2(p_nz)))


@dataclass(frozen=True, slots=False)
class QualityGate:
    """Hard pass/fail gate.

    Defaults reject pixel-flat frames (all-black, all-white, single colour)
    and frames with too little luma spread to carry useful information for
    downstream models. Boundaries are inclusive.
    """

    min_mean_luma: float = 15.0 / 255.0
    max_mean_luma: float = 240.0 / 255.0
    min_luma_variance: float = 5.0

    def passes(self, metrics: QualityMetrics) -> bool:
        return (
            self.min_mean_luma <= metrics.mean_luma <= self.max_mean_luma
            and metrics.luma_variance >= self.min_luma_variance
        )


def compute_quality(rgb: RgbArray, saliency: float | None = None) -> QualityMetrics:
    """Compute all per-frame quality signals from a packed RGB24 array.

    ``saliency`` is the optional Apple Vision attention mass for this frame.
    When ``None``, ``QualityMetrics.saliency_mass`` is reported as ``0.0``.
    """
    luma = rgb_to_luma(rgb)
    return QualityMetrics(
        laplacian_var=laplacian_variance(luma),
        mean_luma=mean_luma(luma),
        luma_variance=luma_variance(luma),
        entropy=entropy(luma),
        saliency_mass=0.0 if saliency is None else float(saliency),
    )
