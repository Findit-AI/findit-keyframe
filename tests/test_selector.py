"""Tests for the stratified-bucket selection logic.

These tests are decode-free: we construct ``FrameCandidate`` objects
synthetically so the algorithm under test is isolated.
"""

from __future__ import annotations

import numpy as np

from findit_keyframe.decode import FrameCandidate
from findit_keyframe.selector import compute_n_buckets, select_keyframes
from findit_keyframe.types import Config, Shot


def _make_rgb(sharpness_level: str, size: int = 64) -> np.ndarray:
    """Make an RGB image with a known approximate sharpness.

    Fixtures deliberately sit inside the default quality gates (mid-grey
    backgrounds, no 100% clipping) so the test exercises the *sharpness*
    path rather than tripping black/bright/clipping filters.

    Arguments:
        sharpness_level: ``'sharp'``, ``'blurry'``, ``'black'``, ``'bright'``,
            or ``'flat'``.
    """
    if sharpness_level == "sharp":
        # Mid-grey background with a dense grid of darker/brighter lines —
        # high Sobel response, no clipping, no flat gate.
        img = np.full((size, size, 3), 128, dtype=np.uint8)
        img[::4, :, :] = 200    # brighter horizontal lines every 4 rows
        img[:, ::4, :] = 60     # darker vertical lines every 4 cols
        return img
    if sharpness_level == "blurry":
        # Soft ramp confined to 60–200 — no clipping, low gradient density.
        ramp = np.linspace(60, 200, size, dtype=np.uint8)
        img = np.broadcast_to(ramp, (size, size)).copy()
        return np.stack([img, img, img], axis=-1)
    if sharpness_level == "black":
        return np.zeros((size, size, 3), dtype=np.uint8)
    if sharpness_level == "bright":
        return np.full((size, size, 3), 250, dtype=np.uint8)
    if sharpness_level == "flat":
        return np.full((size, size, 3), 128, dtype=np.uint8)
    raise ValueError(sharpness_level)


# ---------- compute_n_buckets --------------------------------------------------


def test_n_buckets_short_shot_rounds_up() -> None:
    # 3 s shot @ target_interval=4 → ceil(3/4) = 1 bucket.
    shot = Shot(start_sec=0.0, end_sec=3.0)
    assert compute_n_buckets(shot, Config()) == 1


def test_n_buckets_exact_multiple() -> None:
    # 12 s @ target_interval=4 → exactly 3 buckets.
    shot = Shot(start_sec=0.0, end_sec=12.0)
    assert compute_n_buckets(shot, Config()) == 3


def test_n_buckets_capped_at_max() -> None:
    # 100 s @ target_interval=4 would be 25, but max_frames=16 caps it.
    shot = Shot(start_sec=0.0, end_sec=100.0)
    assert compute_n_buckets(shot, Config()) == 16


def test_n_buckets_minimum_one() -> None:
    # Even a sub-second shot yields one bucket.
    shot = Shot(start_sec=0.0, end_sec=0.1)
    assert compute_n_buckets(shot, Config()) == 1


def test_n_buckets_custom_interval() -> None:
    shot = Shot(start_sec=0.0, end_sec=20.0)
    cfg = Config(target_interval_sec=5.0)
    assert compute_n_buckets(shot, cfg) == 4


# ---------- select_keyframes ---------------------------------------------------


def test_picks_sharp_over_blurry_within_bucket() -> None:
    """A bucket with one sharp + one blurry candidate picks the sharp one."""
    candidates = [
        FrameCandidate(timestamp_sec=1.0, rgb=_make_rgb("blurry"), bucket_index=0),
        FrameCandidate(timestamp_sec=2.0, rgb=_make_rgb("sharp"), bucket_index=0),
    ]
    picks = select_keyframes(candidates, Config())
    assert len(picks) == 1
    assert picks[0].timestamp_sec == 2.0


def test_one_keyframe_per_bucket() -> None:
    """3 buckets each with 2 candidates → 3 picks, one per bucket."""
    candidates = []
    for bucket_idx in range(3):
        for i in range(2):
            ts = bucket_idx * 4.0 + i * 2.0
            candidates.append(
                FrameCandidate(
                    timestamp_sec=ts,
                    rgb=_make_rgb("sharp" if i == 0 else "blurry"),
                    bucket_index=bucket_idx,
                )
            )
    picks = select_keyframes(candidates, Config())
    assert len(picks) == 3
    # Bucket indices should be 0, 1, 2 in order.
    assert [k.bucket_index for k in picks] == [0, 1, 2]


def test_picks_sorted_by_timestamp() -> None:
    """Output always sorted regardless of input order."""
    candidates = [
        FrameCandidate(timestamp_sec=10.0, rgb=_make_rgb("sharp"), bucket_index=2),
        FrameCandidate(timestamp_sec=5.0, rgb=_make_rgb("sharp"), bucket_index=1),
        FrameCandidate(timestamp_sec=1.0, rgb=_make_rgb("sharp"), bucket_index=0),
    ]
    picks = select_keyframes(candidates, Config())
    timestamps = [k.timestamp_sec for k in picks]
    assert timestamps == sorted(timestamps)


def test_black_candidates_trigger_fallback() -> None:
    """A bucket of only-black candidates still produces one keyframe via fallback."""
    candidates = [
        FrameCandidate(timestamp_sec=t, rgb=_make_rgb("black"), bucket_index=0)
        for t in (0.0, 0.5, 1.0)
    ]
    picks = select_keyframes(candidates, Config())
    # Fallback path keeps one even though all are unusable.
    assert len(picks) == 1


def test_empty_bucket_produces_no_keyframe() -> None:
    assert select_keyframes([], Config()) == []


def test_mixed_bad_and_good_prefers_good() -> None:
    """Good candidates beat fallback candidates when both exist."""
    candidates = [
        FrameCandidate(timestamp_sec=0.0, rgb=_make_rgb("black"), bucket_index=0),
        FrameCandidate(timestamp_sec=0.5, rgb=_make_rgb("sharp"), bucket_index=0),
        FrameCandidate(timestamp_sec=1.0, rgb=_make_rgb("flat"), bucket_index=0),
    ]
    picks = select_keyframes(candidates, Config())
    assert len(picks) == 1
    # The 'sharp' edge image survives filtering; the 'black' and 'flat' ones don't.
    assert picks[0].timestamp_sec == 0.5
