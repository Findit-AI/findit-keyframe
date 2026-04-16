"""Core value types for findit-keyframe.

These types are written for a 1:1 Rust translation. See ``docs/rust-porting.md``
for the Python ↔ Rust field map.

Design rules:

* ``Timebase``, ``Timestamp``, ``ShotRange`` and ``QualityMetrics`` are
  ``frozen=True`` — they are value types whose identity is their content.
* ``Timebase`` and ``Timestamp`` use *semantic* equality (1/2 == 2/4,
  ``1000 @ 1/1000 == 90000 @ 1/90000``) to mirror the upstream
  ``scenesdetect`` Rust crate.
* ``SamplingConfig`` and ``ExtractedKeyframe`` are intentionally mutable so
  callers can tweak knobs and attach downstream metadata before serialising.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

__all__ = [
    "Confidence",
    "ExtractedKeyframe",
    "QualityMetrics",
    "SamplingConfig",
    "ShotRange",
    "Timebase",
    "Timestamp",
]


# --------------------------------------------------------------------------- #
# Timebase                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False, slots=False)
class Timebase:
    """Rational timebase ``num / den`` measured in seconds-per-tick.

    Mirrors ``scenesdetect::frame::Timebase``: the denominator is strictly
    positive (``> 0``) and equality is *value-based* — ``Timebase(1, 2)`` and
    ``Timebase(2, 4)`` compare equal and hash identically.

    Args:
        num: Numerator. May be zero (degenerate "always now" timebase) but
            cannot be negative under normal use.
        den: Denominator. Must be strictly positive.

    Raises:
        ValueError: If ``den <= 0``.
    """

    num: int
    den: int

    def __post_init__(self) -> None:
        if self.den <= 0:
            raise ValueError(f"Timebase den must be > 0, got {self.den}")

    # Equality and hashing are reduced-form: we collapse via gcd so that
    # 1/2, 2/4, 5/10 all hash and compare the same way.
    def _reduced(self) -> tuple[int, int]:
        g = math.gcd(abs(self.num), self.den)
        if g == 0:
            return (0, self.den)
        return (self.num // g, self.den // g)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timebase):
            return NotImplemented
        return self._reduced() == other._reduced()

    def __hash__(self) -> int:
        return hash(self._reduced())

    def __repr__(self) -> str:
        return f"Timebase({self.num}/{self.den})"


# --------------------------------------------------------------------------- #
# Timestamp                                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False, slots=False)
class Timestamp:
    """A point in time at a given timebase.

    The wall-clock value in seconds is ``pts * timebase.num / timebase.den``.
    Equality and ordering are *semantic* — comparisons across different
    timebases use exact integer cross-multiplication, so
    ``Timestamp(1000, 1/1000) == Timestamp(90000, 1/90000)``.
    """

    pts: int
    timebase: Timebase

    @property
    def seconds(self) -> float:
        """Wall-clock value as a 64-bit float."""
        return self.pts * self.timebase.num / self.timebase.den

    # ----- semantic comparison via cross-multiply (no float loss) ---------- #

    def _key(self) -> tuple[int, int]:
        """Return ``(numerator, denominator)`` of ``pts * num / den`` reduced."""
        n = self.pts * self.timebase.num
        d = self.timebase.den
        g = math.gcd(abs(n), d)
        if g == 0:
            return (0, d)
        return (n // g, d // g)

    def _cross(self, other: Timestamp) -> tuple[int, int]:
        """Return ``(self_scaled, other_scaled)`` over a common denominator."""
        return (
            self.pts * self.timebase.num * other.timebase.den,
            other.pts * other.timebase.num * self.timebase.den,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Timestamp):
            return NotImplemented
        a, b = self._cross(other)
        return a == b

    def __lt__(self, other: Timestamp) -> bool:
        a, b = self._cross(other)
        return a < b

    def __le__(self, other: Timestamp) -> bool:
        a, b = self._cross(other)
        return a <= b

    def __gt__(self, other: Timestamp) -> bool:
        a, b = self._cross(other)
        return a > b

    def __ge__(self, other: Timestamp) -> bool:
        a, b = self._cross(other)
        return a >= b

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return f"Timestamp(pts={self.pts}, {self.timebase!r}, seconds={self.seconds:g})"


# --------------------------------------------------------------------------- #
# ShotRange                                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=False)
class ShotRange:
    """Half-open shot interval ``[start, end)``.

    ``start`` and ``end`` may use different timebases — the
    ``Timestamp`` semantic comparison handles the mixed case correctly.

    Raises:
        ValueError: If ``end <= start``. Zero-duration shots are a sign of
            upstream input corruption and are rejected at construction.
    """

    start: Timestamp
    end: Timestamp

    def __post_init__(self) -> None:
        if not (self.start < self.end):
            raise ValueError(
                f"ShotRange end must be strictly after start: start={self.start!r}, "
                f"end={self.end!r}"
            )

    @property
    def duration_sec(self) -> float:
        """Shot length in seconds (``end - start``)."""
        return self.end.seconds - self.start.seconds


# --------------------------------------------------------------------------- #
# Confidence                                                                  #
# --------------------------------------------------------------------------- #


class Confidence(enum.StrEnum):
    """Per-keyframe confidence tag, surfaced in the manifest output.

    * ``High`` — selected from its native bin's quality-gated pool.
    * ``Low`` — selected from an expanded fallback window (adjacent bins).
    * ``Degraded`` — all candidates failed the hard gate; force-picked the
      best of a bad lot.
    """

    High = "high"
    Low = "low"
    Degraded = "degraded"


# --------------------------------------------------------------------------- #
# SamplingConfig                                                              #
# --------------------------------------------------------------------------- #


@dataclass(slots=False)
class SamplingConfig:
    """User-tunable knobs controlling stratified temporal sampling.

    Defaults are documented in ``docs/algorithm.md`` §7 ("Parameter
    Rationale") and chosen for 24-60 fps source video where each shot is
    between roughly 0.5 s and several minutes long.
    """

    target_interval_sec: float = 4.0
    candidates_per_bin: int = 6
    max_frames_per_shot: int = 16
    boundary_shrink_pct: float = 0.02
    fallback_expand_pct: float = 0.20
    target_size: int = 384


# --------------------------------------------------------------------------- #
# QualityMetrics                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=False)
class QualityMetrics:
    """Per-frame quality signals computed by ``findit_keyframe.quality``.

    All fields are floats:

    * ``laplacian_var`` — variance of a 3x3 Laplacian-filtered luma image,
      a sharpness/blur proxy. Higher is sharper.
    * ``mean_luma`` — mean luminance normalised to ``[0.0, 1.0]``.
    * ``luma_variance`` — sample variance of luma values (raw, on the 0-255
      integer scale before normalisation).
    * ``entropy`` — Shannon entropy in bits of the 256-bin luma histogram.
    * ``saliency_mass`` — Apple Vision attention score in ``[0.0, 1.0]``;
      ``0.0`` when no saliency provider is configured. The bundled
      :class:`findit_keyframe.saliency.AppleVisionSaliencyProvider` derives
      this as ``clamp(sum(area * confidence), 0, 1)`` over the request's
      ``salientObjects`` bounding boxes, *not* from the heatmap
      ``CVPixelBuffer`` — see that module's docstring for rationale.
    """

    laplacian_var: float
    mean_luma: float
    luma_variance: float
    entropy: float
    saliency_mass: float


# --------------------------------------------------------------------------- #
# ExtractedKeyframe                                                           #
# --------------------------------------------------------------------------- #


@dataclass(slots=False)
class ExtractedKeyframe:
    """One keyframe selected for a shot, with raw RGB pixels and metadata.

    ``rgb`` is a packed RGB24 byte buffer of length ``width * height * 3``.
    The buffer is intentionally ``bytes`` (not ``np.ndarray``) so the
    contract is portable and so the Rust port can use ``Vec<u8>`` directly.
    """

    shot_id: int
    timestamp: Timestamp
    bucket_index: int
    rgb: bytes
    width: int
    height: int
    quality: QualityMetrics
    confidence: Confidence
