"""Stratified temporal bucket selection.

Algorithm:

    1. ``N = clamp(ceil(duration / target_interval), 1, max_frames_per_shot)``
    2. For each of the N buckets, score every candidate in a single pass
       via :func:`quality.score_frame`.
    3. Hard filter: drop unusable (black / bright / flat) and below-minimum
       sharpness.
    4. Pick ``argmax(sharpness)`` among survivors.
    5. Fallback: if all candidates were filtered out, pick the sharpest
       anyway — we prefer a degraded frame to a gap in temporal coverage.

Zero redundant work: each candidate is scored exactly once. The filter,
the ranker, and the final :class:`Keyframe` all consume the same cached
:class:`FrameScore`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PIL import Image

from .quality import FrameScore, score_frame
from .types import Config, Keyframe, Shot

if TYPE_CHECKING:
    from .decode import FrameCandidate


def compute_n_buckets(shot: Shot, config: Config) -> int:
    """Return the number of time buckets for this shot.

    ``N = clamp(ceil(duration / target_interval), 1, max_frames_per_shot)``
    """
    n = math.ceil(shot.duration_sec / config.target_interval_sec)
    return max(1, min(n, config.max_frames_per_shot))


def select_keyframes(
    candidates: list[FrameCandidate],
    config: Config,
) -> list[Keyframe]:
    """Produce one keyframe per non-empty bucket, sorted by timestamp."""
    by_bucket: dict[int, list[FrameCandidate]] = {}
    for c in candidates:
        by_bucket.setdefault(c.bucket_index, []).append(c)

    picks: list[Keyframe] = []
    for bucket_idx in sorted(by_bucket.keys()):
        pick = _select_from_bucket(by_bucket[bucket_idx], config, bucket_idx)
        if pick is not None:
            picks.append(pick)

    picks.sort(key=lambda k: k.timestamp_sec)
    return picks


def _select_from_bucket(
    bucket: list[FrameCandidate],
    config: Config,
    bucket_idx: int,
) -> Keyframe | None:
    """Pick the best candidate in this bucket.

    Two-pass selection over a single set of scores:

    * **Strict**: apply hard gates and the sharpness floor; take argmax.
    * **Fallback**: if strict was empty, take argmax over everything so
      the bucket still contributes a keyframe.

    The image of the chosen frame is materialized to ``PIL.Image`` exactly
    once, at the end.
    """
    if not bucket:
        return None

    # One score per candidate — no repeated downscale / cvtColor / stats.
    scored: list[tuple[FrameCandidate, FrameScore]] = [
        (cand, score_frame(cand.rgb)) for cand in bucket
    ]

    # Strict pass: hard gates + sharpness floor.
    strict = [
        (cand, score)
        for cand, score in scored
        if not score.is_unusable(
            config.black_mean_threshold,
            config.bright_mean_threshold,
            config.luma_variance_threshold,
            config.sat_variance_threshold,
            config.max_clipping,
        )
        and score.sharpness >= config.min_sharpness
    ]

    pool = strict if strict else scored
    best_cand, best_score = max(pool, key=lambda pair: pair[1].sharpness)

    image = Image.fromarray(best_cand.rgb, mode="RGB")
    return Keyframe(
        timestamp_sec=best_cand.timestamp_sec,
        image=image,
        sharpness=best_score.sharpness,
        brightness=best_score.brightness,
        bucket_index=bucket_idx,
    )


__all__ = ["compute_n_buckets", "select_keyframes"]
