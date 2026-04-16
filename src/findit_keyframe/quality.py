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
    """Convert packed RGB24 to BT.601 limited-range luma (``Y ∈ [16, 235]``).

    Formula: ``((66*R + 129*G + 25*B + 128) >> 8) + 16``. The integer
    fixed-point form is identical to scenesdetect's so a Y-plane round-trip
    is bit-exact across crates.

    Args:
        rgb: Array of shape ``(H, W, 3)`` and dtype ``uint8``. Channel
            order is RGB.

    Returns:
        Array of shape ``(H, W)`` and dtype ``uint8`` carrying the luma plane.

    Raises:
        ValueError: If ``rgb`` is not ``uint8`` or its shape is not
            ``(H, W, 3)``.
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
    """Variance of a 3x3 Laplacian-filtered luma image; a sharpness proxy.

    Kernel ``[[0, 1, 0], [1, -4, 1], [0, 1, 0]]``. The one-pixel border is
    dropped (no padding). Variance uses the population denominator (``N``)
    to match OpenCV's ``meanStdDev(Laplacian(...))``.

    Args:
        luma: Single-plane luma array, dtype ``uint8``.

    Returns:
        Population variance of the filtered interior pixels, as a Python
        ``float``. Higher values mean sharper content.

    Raises:
        ValueError: If ``luma`` is not 2-D or smaller than ``3x3``.
    """
    if luma.ndim != 2:
        raise ValueError(f"luma must be 2D, got shape {luma.shape}")
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        raise ValueError(f"luma too small for 3x3 Laplacian: {luma.shape}")
    f = luma.astype(np.int32)
    out = f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:] - 4 * f[1:-1, 1:-1]
    return float(out.var())


def mean_luma(luma: LumaArray) -> float:
    """Compute the arithmetic mean of luma values normalised to ``[0.0, 1.0]``.

    Args:
        luma: Single-plane luma array on the 0-255 scale.

    Returns:
        ``float(luma.mean()) / 255.0``.
    """
    return float(luma.mean()) / 255.0


def luma_variance(luma: LumaArray) -> float:
    """Compute the sample variance (``ddof=1``) of luma on the raw 0-255 scale.

    Args:
        luma: Single-plane luma array on the 0-255 scale.

    Returns:
        Sample variance with Bessel's correction. Pixel-flat frames return
        ``0.0``.
    """
    return float(luma.var(ddof=1))


def entropy(luma: LumaArray, bins: int = 256) -> float:
    """Compute the Shannon entropy in bits of a ``bins``-bin luma histogram.

    Histogram range is fixed to ``[0, 256)`` (i.e. one bin per integer level
    when ``bins == 256``), making the function input-only deterministic.

    Args:
        luma: Single-plane luma array on the 0-255 scale.
        bins: Number of histogram bins. Defaults to ``256``.

    Returns:
        ``-Σ p_i * log2(p_i)`` over non-zero probabilities. Range is
        ``[0, log2(bins)]``; ``0.0`` for delta distributions and empty input.
    """
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
        """Return ``True`` when ``metrics`` clears every threshold (boundaries inclusive).

        Args:
            metrics: Per-frame metrics produced by :func:`compute_quality`.

        Returns:
            ``True`` iff ``mean_luma`` lies inclusively in
            ``[min_mean_luma, max_mean_luma]`` *and* ``luma_variance ≥
            min_luma_variance``.
        """
        return (
            self.min_mean_luma <= metrics.mean_luma <= self.max_mean_luma
            and metrics.luma_variance >= self.min_luma_variance
        )


def compute_quality(rgb: RgbArray, saliency: float | None = None) -> QualityMetrics:
    """Compute all per-frame quality signals from a packed RGB24 array.

    Args:
        rgb: Array of shape ``(H, W, 3)`` and dtype ``uint8``; channel order RGB.
        saliency: Optional saliency mass in ``[0.0, 1.0]`` for this frame.
            ``None`` (the default) records ``0.0`` so frames extracted
            without a saliency provider remain comparable.

    Returns:
        A :class:`QualityMetrics` populated with all five signals.

    Raises:
        ValueError: Propagated from :func:`rgb_to_luma` when the input has
            the wrong shape or dtype.
    """
    luma = rgb_to_luma(rgb)
    return QualityMetrics(
        laplacian_var=laplacian_variance(luma),
        mean_luma=mean_luma(luma),
        luma_variance=luma_variance(luma),
        entropy=entropy(luma),
        saliency_mass=0.0 if saliency is None else float(saliency),
    )
