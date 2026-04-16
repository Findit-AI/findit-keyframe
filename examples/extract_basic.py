#!/usr/bin/env python3
"""End-to-end programmatic example for ``findit-keyframe``.

Demonstrates the public Python API as an alternative to the
``findit-keyframe extract`` CLI. Run from the repository root::

    python examples/extract_basic.py path/to/video.mp4

The script:

1. Opens the video with :class:`VideoDecoder`.
2. Builds two synthetic back-to-back shots that cover the whole video.
   In real use, shot boundaries come from
   `scenesdetect <https://github.com/Findit-AI/scenesdetect>`_'s output.
3. Runs :func:`extract_all` with default :class:`SamplingConfig` plus the
   platform's preferred :class:`SaliencyProvider` (Apple Vision on macOS
   when ``[macos]`` extras are installed; ``Noop`` elsewhere).
4. Prints the chosen keyframe's bin index, timestamp, confidence, and
   Laplacian variance for each shot.

Use this as a template for wiring ``findit-keyframe`` into a larger
pipeline that already has shot boundaries and wants in-process keyframe
extraction without going through the CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

from findit_keyframe import (
    SamplingConfig,
    ShotRange,
    Timebase,
    Timestamp,
    VideoDecoder,
    default_saliency_provider,
    extract_all,
)


def _ts(seconds: float, timebase: Timebase) -> Timestamp:
    """Build a Timestamp for ``seconds`` in the decoder's native timebase."""
    return Timestamp(round(seconds * timebase.den / timebase.num), timebase)


def _build_demo_shots(decoder: VideoDecoder) -> list[ShotRange]:
    """Two back-to-back shots covering the whole video.

    A real pipeline would replace this with the upstream shot list (for
    example, by calling :func:`findit_keyframe.cli._parse_shot_json` on
    a scenesdetect output file).
    """
    duration = decoder.duration_sec
    if duration <= 0.0:
        raise ValueError(f"video reports unknown duration ({duration})")
    midpoint = duration / 2.0
    return [
        ShotRange(start=_ts(0.0, decoder.timebase), end=_ts(midpoint, decoder.timebase)),
        ShotRange(start=_ts(midpoint, decoder.timebase), end=_ts(duration, decoder.timebase)),
    ]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} VIDEO_PATH", file=sys.stderr)
        return 1

    video_path = Path(sys.argv[1])
    if not video_path.is_file():
        print(f"error: {video_path} not found", file=sys.stderr)
        return 1

    config = SamplingConfig()
    saliency = default_saliency_provider()
    print(f"saliency provider: {type(saliency).__name__}", file=sys.stderr)

    with VideoDecoder.open(video_path, target_size=config.target_size) as decoder:
        shots = _build_demo_shots(decoder)
        keyframes_per_shot = extract_all(shots, decoder, config, saliency_provider=saliency)

    for shot_id, shot_keyframes in enumerate(keyframes_per_shot):
        print(f"shot {shot_id}: {len(shot_keyframes)} keyframe(s)")
        for kf in shot_keyframes:
            print(
                f"  bin={kf.bucket_index}  "
                f"t={kf.timestamp.seconds:7.3f}s  "
                f"conf={kf.confidence.value:<8}  "
                f"laplacian={kf.quality.laplacian_var:9.1f}  "
                f"saliency={kf.quality.saliency_mass:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
