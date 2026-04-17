"""Public data types.

Kept deliberately small: three dataclasses, no hidden state, no methods that
can panic at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True, slots=True)
class Shot:
    """A shot's half-open time range in seconds.

    Attributes:
        start_sec: First timestamp of the shot (inclusive).
        end_sec: First timestamp NOT in the shot (exclusive).
    """

    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        if self.end_sec <= self.start_sec:
            raise ValueError(
                f"Shot end_sec ({self.end_sec}) must be > start_sec ({self.start_sec})"
            )

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True, slots=True)
class Keyframe:
    """One extracted keyframe, ready to hand to a VLM.

    Attributes:
        timestamp_sec: PTS of the frame in seconds (from video start).
        image: RGB PIL Image. HuggingFace processors consume this directly.
        sharpness: Tenengrad score. Higher is sharper.
        brightness: Mean Y (luma) in 0-255.
        bucket_index: Which time bucket of the shot produced this frame.
    """

    timestamp_sec: float
    image: Image
    sharpness: float
    brightness: float
    bucket_index: int

    def __repr__(self) -> str:
        return (
            f"Keyframe(t={self.timestamp_sec:.3f}s, "
            f"size={self.image.size}, "
            f"sharpness={self.sharpness:.1f}, "
            f"brightness={self.brightness:.1f}, "
            f"bucket={self.bucket_index})"
        )


@dataclass(frozen=True, slots=True)
class Config:
    """Tuning knobs for keyframe extraction.

    Defaults are tuned for general VLM-description use (SigLIP 2 / Qwen3-VL).
    Override specific fields via keyword arguments.

    Temporal sampling:
        target_interval_sec: Aim for one keyframe every N seconds within a shot.
            A 12 s shot with ``target_interval_sec=4`` yields 3 buckets.
        max_frames_per_shot: Hard cap to respect VLM token budgets.
        candidates_per_bucket: How many candidate frames to decode per bucket.
            The best one (by sharpness) wins. Higher = more robust selection,
            marginally slower.
        margin_ratio: Fraction of shot duration to skip at both ends.
            Protects against dissolve tails and scene-detector ±2-frame error.

    Quality scoring:
        min_sharpness: Tenengrad floor. Computed at
            :data:`quality.QUALITY_TARGET_DIM` (384 px longest side). Frames
            below are rejected in the strict pass; fallback path selects the
            best-scoring frame anyway. Calibrated empirically:

                * ``< 100``: clearly blurry / motion-blurred
                * ``100–500``: soft but usable
                * ``500+``: sharp

    Hard quality gates (any trip → frame rejected):
        black_mean_threshold: Luma mean below → reject as black frame.
        bright_mean_threshold: Luma mean above → reject as overexposed.
        luma_variance_threshold: Luma variance floor. Combined with
            ``sat_variance_threshold`` via logical AND — a frame is flagged
            flat only when **both** variances are low, so equiluminant
            multi-color frames (same brightness, different hue) are correctly
            kept.
        sat_variance_threshold: HSV saturation variance floor, paired with
            ``luma_variance_threshold``.
        max_clipping: Max tolerated fraction of clipped pixels
            (``max(R, G, B) < 5`` or ``> 250``). Rejects blown-highlight or
            crushed-shadow frames that the luma mean alone wouldn't catch.
            ``0.50`` rejects only egregious cases (fade-to-white, heavy
            overexposure); raise to be stricter.
    """

    # Temporal sampling
    target_interval_sec: float = 4.0
    max_frames_per_shot: int = 16
    candidates_per_bucket: int = 6
    margin_ratio: float = 0.02

    # Sharpness floor
    min_sharpness: float = 100.0

    # Hard quality gates
    black_mean_threshold: float = 15.0
    bright_mean_threshold: float = 240.0
    luma_variance_threshold: float = 5.0
    sat_variance_threshold: float = 3.0
    max_clipping: float = 0.50

    def __post_init__(self) -> None:
        if self.target_interval_sec <= 0:
            raise ValueError("target_interval_sec must be > 0")
        if self.max_frames_per_shot < 1:
            raise ValueError("max_frames_per_shot must be >= 1")
        if self.candidates_per_bucket < 1:
            raise ValueError("candidates_per_bucket must be >= 1")
        if not 0.0 <= self.margin_ratio < 0.5:
            raise ValueError("margin_ratio must be in [0.0, 0.5)")
        if not 0.0 <= self.max_clipping <= 1.0:
            raise ValueError("max_clipping must be in [0.0, 1.0]")
