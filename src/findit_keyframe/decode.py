"""PyAV-backed frame decoding with target-timestamp sampling.

Key design decisions:
    * **Single container, per-shot seek**: caller opens the video once in
      ``extractor.extract``; we seek to each shot's start here. Seeks use
      ``any_frame=False`` to land on the nearest prior keyframe, then we
      decode forward. Sequential decoding inside a shot is cheap.
    * **Target-timestamp sampling**: instead of keeping every decoded frame
      (which could be hundreds of MB per shot at 1080p), we pre-compute a
      list of target timestamps uniformly spaced within each bucket and
      claim the *closest* decoded frame to each target. Memory peak is
      ``n_buckets × candidates_per_bucket`` frames per shot.
    * **Stream thread pool**: ``stream.thread_type = 'AUTO'`` lets libav
      parallelize decode across cores. This is a one-line free ~2x speedup
      on modern H.264.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import av.container
    import av.video.stream
    import numpy.typing as npt

    from .types import Config, Shot


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    """A decoded candidate frame awaiting quality scoring.

    Internal to the extraction pipeline; not part of the public API.
    """

    timestamp_sec: float
    rgb: npt.NDArray[np.uint8]
    bucket_index: int


def _compute_target_timestamps(
    shot: Shot,
    n_buckets: int,
    config: Config,
) -> list[tuple[float, int]]:
    """Compute (target_ts, bucket_index) pairs for candidate sampling.

    Each bucket gets ``candidates_per_bucket`` uniformly-spaced targets,
    shrunk by ``margin_ratio`` at the first and last bucket to avoid
    scene-detector edge artifacts (dissolve tails, ±2-frame uncertainty).
    """
    margin = shot.duration_sec * config.margin_ratio
    bucket_duration = shot.duration_sec / n_buckets
    k = config.candidates_per_bucket

    targets: list[tuple[float, int]] = []
    for b in range(n_buckets):
        b_start = shot.start_sec + b * bucket_duration
        b_end = b_start + bucket_duration

        # Edge buckets get their interior shrunk by ``margin``; interior
        # buckets use the full bucket width.
        if b == 0:
            b_start = max(b_start, shot.start_sec + margin)
        if b == n_buckets - 1:
            b_end = min(b_end, shot.end_sec - margin)

        # Degenerate case: margin ate the whole bucket. Sample the midpoint
        # of the original (un-shrunk) bucket so we still produce a candidate.
        if b_end <= b_start:
            mid = shot.start_sec + (b + 0.5) * bucket_duration
            targets.append((mid, b))
            continue

        width = b_end - b_start
        for i in range(k):
            # Centered offsets: (i + 0.5) / k ∈ {1/2k, 3/2k, …, (2k−1)/2k}
            frac = (i + 0.5) / k
            targets.append((b_start + frac * width, b))

    # Sort by timestamp — the consumer walks these in order alongside the
    # decode stream.
    targets.sort(key=lambda t: t[0])
    return targets


def decode_shot_candidates(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    shot: Shot,
    n_buckets: int,
    config: Config,
) -> list[FrameCandidate]:
    """Decode the shot, claiming the closest frame to each target timestamp.

    Algorithm (single pass over the shot's frames):
        1. Seek to nearest keyframe <= shot.start_sec.
        2. Walk decoded frames in PTS order, keeping (prev_frame, cur_frame).
        3. For each remaining target timestamp, the closest of
           {prev_frame, cur_frame} is claimed once ``cur_ts > target_ts``.
        4. Stop when all targets claimed OR we pass shot.end_sec.

    The "pick closer of prev/cur" trick matters: naively taking the first
    frame past the target can miss by up to a full frame interval, which on
    24 fps content is ~42 ms — noticeable on short buckets.

    Returns:
        Candidates with their bucket_index, ordered by timestamp.
    """
    targets = _compute_target_timestamps(shot, n_buckets, config)
    if not targets:
        return []

    time_base = stream.time_base
    if time_base is None:
        raise RuntimeError("video stream has no time_base; cannot decode by PTS")

    # Seek to nearest keyframe <= shot start. PyAV expects PTS in stream's
    # timebase. ``any_frame=False`` forces keyframe seek (required for
    # correct B/P frame decoding afterward).
    seek_pts = int(shot.start_sec / float(time_base))
    container.seek(seek_pts, stream=stream, any_frame=False, backward=True)

    candidates: list[FrameCandidate] = []
    target_idx = 0
    n_targets = len(targets)

    # Previous frame's (ts, rgb) to enable "pick closer" logic.
    prev: tuple[float, npt.NDArray[np.uint8]] | None = None

    for frame in container.decode(stream):
        if frame.pts is None:
            # VFR edge case / corrupt stream. Skip quietly.
            continue
        ts = float(frame.pts * time_base)

        # Frames before the shot (due to keyframe seek overshoot) are
        # reference material only.
        if ts < shot.start_sec - 1e-6:
            prev = None
            continue

        # Past the shot boundary — flush remaining targets with prev if any,
        # then stop.
        if ts > shot.end_sec + 1e-6:
            while target_idx < n_targets and prev is not None:
                target_ts, bucket = targets[target_idx]
                candidates.append(
                    FrameCandidate(timestamp_sec=prev[0], rgb=prev[1], bucket_index=bucket)
                )
                target_idx += 1
            break

        # Materialize the current frame as an RGB ndarray. ``to_ndarray``
        # with format='rgb24' forces swscale to RGB, matching PIL.Image and
        # most VLM preprocessors.
        rgb = frame.to_ndarray(format="rgb24")

        # Claim every target we've now passed.
        while target_idx < n_targets:
            target_ts, bucket = targets[target_idx]
            if ts < target_ts:
                break  # cur frame still before target — wait for next

            # Decide whether prev or cur is closer to target_ts.
            if prev is not None and abs(prev[0] - target_ts) < abs(ts - target_ts):
                chosen_ts, chosen_rgb = prev
            else:
                chosen_ts, chosen_rgb = ts, rgb
            candidates.append(
                FrameCandidate(timestamp_sec=chosen_ts, rgb=chosen_rgb, bucket_index=bucket)
            )
            target_idx += 1

        if target_idx >= n_targets:
            break  # all claimed, no need to keep decoding this shot

        prev = (ts, rgb)

    return candidates


__all__ = ["FrameCandidate", "decode_shot_candidates"]
