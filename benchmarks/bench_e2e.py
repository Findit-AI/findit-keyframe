#!/usr/bin/env python3
"""End-to-end keyframe extraction benchmark.

Usage::

    python benchmarks/bench_e2e.py --video PATH [--shots PATH] [--target-size N]
                                   [--results-md PATH] [--quiet]

If ``--shots`` is omitted, uniform 4-second shots covering the whole video
are generated. Each run prints a JSON summary on stdout (suppressed with
``--quiet``) and appends a Markdown row to ``benchmarks/results.md`` with
date, git SHA, and the headline numbers.

The Rust port (P5+) must beat this baseline by 5-10x on the same input;
keep this script self-contained so identical fixtures can be replayed
across implementations.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from findit_keyframe.cli import _parse_shot_json
from findit_keyframe.decoder import VideoDecoder
from findit_keyframe.sampler import extract_all
from findit_keyframe.types import SamplingConfig, ShotRange, Timebase, Timestamp


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _uniform_shots(duration_sec: float, interval_sec: float = 4.0) -> list[ShotRange]:
    """Synthetic shot list covering ``[0, duration_sec)`` in equal slices."""
    tb = Timebase(1, 1000)
    n = max(1, int(duration_sec // interval_sec))
    return [
        ShotRange(
            start=Timestamp(round(i * interval_sec * 1000), tb),
            end=Timestamp(round(min((i + 1) * interval_sec, duration_sec) * 1000), tb),
        )
        for i in range(n)
    ]


def _peak_memory_mb() -> float:
    """Process peak RSS in MB. Linux reports KB; macOS reports bytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return rss / divisor


def run_benchmark(
    video: Path,
    shots: list[ShotRange] | None = None,
    target_size: int = 384,
) -> dict[str, Any]:
    """Time ``extract_all`` on ``video`` and return a result dict."""
    config = SamplingConfig(target_size=target_size)
    with VideoDecoder.open(video, target_size=config.target_size) as decoder:
        all_shots = shots if shots is not None else _uniform_shots(decoder.duration_sec)
        t0 = time.perf_counter()
        keyframes_per_shot = extract_all(all_shots, decoder, config)
        wall = time.perf_counter() - t0
        duration = decoder.duration_sec
    n_keyframes = sum(len(s) for s in keyframes_per_shot)
    return {
        "video": str(video),
        "duration_sec": round(duration, 3),
        "shots": len(all_shots),
        "keyframes": n_keyframes,
        "wall_sec": round(wall, 3),
        "kf_per_sec": round(n_keyframes / wall, 1) if wall > 0 else None,
        "memory_mb": round(_peak_memory_mb(), 1),
        "target_size": target_size,
    }


_RESULTS_HEADER = (
    "# findit-keyframe benchmarks\n"
    "\n"
    "Append-only log of `bench_e2e.py` runs. Each row is one run.\n"
    "\n"
    "| Date (UTC) | Git | Video | Duration (s) | Shots | Keyframes "
    "| Wall (s) | KF/s | Mem (MB) | Target |\n"
    "|------------|-----|-------|--------------|-------|-----------"
    "|----------|------|----------|--------|\n"
)


def append_result(results_md: Path, result: dict[str, Any], git_sha: str) -> None:
    if not results_md.exists():
        results_md.write_text(_RESULTS_HEADER)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    row = (
        f"| {timestamp} | `{git_sha}` | `{Path(result['video']).name}` "
        f"| {result['duration_sec']} | {result['shots']} | {result['keyframes']} "
        f"| {result['wall_sec']} | {result['kf_per_sec']} | {result['memory_mb']} "
        f"| {result['target_size']} |\n"
    )
    with results_md.open("a") as f:
        f.write(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bench_e2e",
        description="End-to-end findit-keyframe extraction benchmark.",
    )
    parser.add_argument("--video", type=Path, required=True, help="Source video file.")
    parser.add_argument(
        "--shots",
        type=Path,
        default=None,
        help="Optional shot JSON; defaults to 4-second uniform shots.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=384,
        help="Output frame edge length (default 384, matching SamplingConfig).",
    )
    parser.add_argument(
        "--results-md",
        type=Path,
        default=Path(__file__).parent / "results.md",
        help="Markdown results file to append to (default: benchmarks/results.md).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout JSON summary.")
    args = parser.parse_args()

    shots = _parse_shot_json(args.shots) if args.shots is not None else None
    result = run_benchmark(args.video, shots, target_size=args.target_size)
    sha = _git_sha()
    append_result(args.results_md, result, sha)

    if not args.quiet:
        print(json.dumps({**result, "git_sha": sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
