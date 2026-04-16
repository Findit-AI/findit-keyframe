"""findit-keyframe: per-shot keyframe extraction with stratified temporal sampling.

Public API surface re-exports the user-facing types and top-level functions.
Internal helpers live in their respective modules and are not part of the
stable API.
"""

from __future__ import annotations

from findit_keyframe.decoder import (
    DecodedFrame,
    Strategy,
    VideoDecoder,
    pick_strategy,
)
from findit_keyframe.quality import QualityGate, compute_quality
from findit_keyframe.saliency import (
    AppleVisionSaliencyProvider,
    NoopSaliencyProvider,
    SaliencyProvider,
    default_saliency_provider,
)
from findit_keyframe.sampler import extract_all, extract_for_shot
from findit_keyframe.types import (
    Confidence,
    ExtractedKeyframe,
    QualityMetrics,
    SamplingConfig,
    ShotRange,
    Timebase,
    Timestamp,
)

__version__ = "0.0.0"

__all__ = [
    "AppleVisionSaliencyProvider",
    "Confidence",
    "DecodedFrame",
    "ExtractedKeyframe",
    "NoopSaliencyProvider",
    "QualityGate",
    "QualityMetrics",
    "SaliencyProvider",
    "SamplingConfig",
    "ShotRange",
    "Strategy",
    "Timebase",
    "Timestamp",
    "VideoDecoder",
    "__version__",
    "compute_quality",
    "default_saliency_provider",
    "extract_all",
    "extract_for_shot",
    "pick_strategy",
]
