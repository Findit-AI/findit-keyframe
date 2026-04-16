"""findit-keyframe: per-shot keyframe extraction with stratified temporal sampling.

Public API surface re-exports the user-facing types and (once implemented)
top-level functions. Internal helpers live in their respective modules and
are not part of the stable API.
"""

from __future__ import annotations

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
    "Confidence",
    "ExtractedKeyframe",
    "QualityMetrics",
    "SamplingConfig",
    "ShotRange",
    "Timebase",
    "Timestamp",
    "__version__",
]
