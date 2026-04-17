"""Stratified temporal bucket selection.

The library's only algorithmic opinion lives here:

    1. Divide a shot into ``N = clamp(ceil(duration / target_interval), 1, max)``
       time buckets.
    2. Within each bucket, sample ``candidates_per_bucket`` candidate frames.
    3. Hard-filter unusable candidates (black / overexposed / solid color).
    4. Among survivors, pick ``argmax(sharpness)``.
    5. Fallback: if all candidates are filtered, pick the sharpest anyway
       so each bucket contributes one keyframe whenever possible.

The fallback matters: if we silently drop buckets the caller ends up with
gaps in time coverage, which defeats the point of stratification.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PIL import Image

from .quality import is_unusable_frame, luma_stats, rgb_to_luma, tenengrad_sharpness

if TYPE_CHECKING:
    from .decode import FrameCandidate
    from .types import Config, Keyframe, Shot

from .types import Keyframe


def compute_n_buckets(shot: Shot, config: Config) -> int:
    """Number of time buckets for the given shot.

    ``N = clamp(ceil(duration / target_interval), 1, max_frames_per_shot)``.
    """
    n = math.ceil(shot.duration_sec / config.target_interval_sec)
    return max(1, min(n, config.max_frames_per_shot))


def select_keyframes(
    candidates: list[FrameCandidate],
    config: Config,
) -> list[Keyframe]:
    """Pick the best keyframe per bucket.

    Input candidates may arrive in any order; output keyframes are sorted
    by ``timestamp_sec`` (monotonically increasing).
    """
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
    """Two-pass selection within one bucket.

    Pass 1: filter unusable (black/bright/flat), require ``sharpness >= min``,
            take argmax.
    Pass 2 (fallback): if Pass 1 empty, take argmax sharpness from all
            candidates regardless of thresholds. Last-resort coverage.
    """
    if not bucket:
        return None

    # Score every candidate once; reuse scores across both passes.
    scored: list[tuple[FrameCandidate, float, float]] = []  # (cand, sharp, brightness)
    for cand in bucket:
        luma = rgb_to_luma(cand.rgb)
        sharp = tenengrad_sharpness(luma)
        mean, _var = luma_stats(luma)
        scored.append((cand, sharp, mean))

    # Pass 1: hard filters + sharpness floor.
    strict: list[tuple[FrameCandidate, float, float]] = []
    for cand, sharp, mean in scored:
        luma = rgb_to_luma(cand.rgb)  # Recomputed once; cached earlier would waste memory.
        if is_unusable_frame(
            luma,
            config.black_mean_threshold,
            config.bright_mean_threshold,
            config.variance_threshold,
        ):
            continue
        if sharp < config.min_sharpness:
            continue
        strict.append((cand, sharp, mean))

    pool = strict if strict else scored
    best = max(pool, key=lambda t: t[1])
    best_cand, best_sharp, best_brightness = best

    # Lazy conversion: we only materialize PIL.Image for the one chosen frame.
    image = Image.fromarray(best_cand.rgb, mode="RGB")
    return Keyframe(
        timestamp_sec=best_cand.timestamp_sec,
        image=image,
        sharpness=best_sharp,
        brightness=best_brightness,
        bucket_index=bucket_idx,
    )


__all__ = ["compute_n_buckets", "select_keyframes"]
