"""Top-level extraction API.

The single public entry point, ``extract``, opens the video once and feeds
each shot through the decode → select pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import av

from .decode import decode_shot_candidates
from .selector import compute_n_buckets, select_keyframes
from .types import Config, Keyframe, Shot


def extract(
    video_path: Path | str,
    shots: Sequence[Shot],
    config: Config | None = None,
) -> list[list[Keyframe]]:
    """Extract temporally-distributed, quality-aware keyframes for each shot.

    Args:
        video_path: Path to the input video file.
        shots: Shot time ranges (in seconds) from an upstream scene detector.
            Must be non-empty for the result to be non-empty.
        config: Tuning knobs. ``None`` → use :class:`Config` defaults tuned
            for VLM description use.

    Returns:
        ``result[i]`` is the list of keyframes selected from ``shots[i]``,
        sorted by timestamp. Empty shots or shots where decoding failed are
        represented by an empty list — the outer list's length always equals
        ``len(shots)``.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist.
        ValueError: If the container has no video stream.

    Example:
        >>> from findit_keyframe import extract, Shot
        >>> shots = [Shot(0.0, 12.3), Shot(12.3, 47.8)]
        >>> kfs_per_shot = extract("input.mp4", shots)
        >>> for kf in kfs_per_shot[0]:
        ...     kf.image.save(f"{kf.timestamp_sec:.2f}.jpg")
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cfg = config if config is not None else Config()
    results: list[list[Keyframe]] = []

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {video_path}")

        stream = container.streams.video[0]
        # libav-side multithreaded decode: free 1.5–2x speedup on H.264/HEVC.
        stream.thread_type = "AUTO"

        for shot in shots:
            keyframes = _extract_single_shot(container, stream, shot, cfg)
            results.append(keyframes)

    return results


def _extract_single_shot(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    shot: Shot,
    config: Config,
) -> list[Keyframe]:
    """Decode + select for one shot. Returns [] on decode error."""
    try:
        n_buckets = compute_n_buckets(shot, config)
        candidates = decode_shot_candidates(container, stream, shot, n_buckets, config)
        return select_keyframes(candidates, config)
    except av.AVError:
        # Corrupt packet / unsupported codec option mid-shot: skip rather
        # than abort the whole batch. Caller sees an empty list for this shot.
        return []


__all__ = ["extract"]
