"""Benchmark keyframe extraction on real videos.

Usage::

    python benchmarks/bench_real_video.py VIDEO_PATH [SHOTS_JSON]

If ``SHOTS_JSON`` is omitted, the whole video is treated as a single shot.
Times the extraction (decode + select) wall-clock per shot, reports an
aggregate throughput, and optionally dumps extracted keyframes to
``./bench_output/``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from findit_keyframe import Config, Shot, extract


def _load_shots(path: Path) -> list[Shot]:
    data = json.loads(path.read_text())
    return [Shot(start_sec=s["start_sec"], end_sec=s["end_sec"]) for s in data["shots"]]


def _whole_video_as_one_shot(video_path: Path) -> list[Shot]:
    """Fallback when no shot JSON provided: ask ffprobe for duration."""
    import subprocess

    ffprobe = Path.home() / ".local/bin/ffprobe"
    if not ffprobe.exists():
        ffprobe_cmd = "ffprobe"
    else:
        ffprobe_cmd = str(ffprobe)
    out = subprocess.run(
        [
            ffprobe_cmd,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(out.stdout.strip())
    return [Shot(start_sec=0.0, end_sec=duration)]


def run(video_path: Path, shots: list[Shot], dump_dir: Path | None = None) -> None:
    print(f"Video:         {video_path}")
    print(f"Shots:         {len(shots)}")
    total_duration = sum(s.duration_sec for s in shots)
    print(f"Total span:    {total_duration:.1f} s")

    cfg = Config()
    print(f"Config:        target_interval={cfg.target_interval_sec}s, "
          f"max_frames={cfg.max_frames_per_shot}, "
          f"candidates_per_bucket={cfg.candidates_per_bucket}")
    print()

    t0 = time.perf_counter()
    result = extract(video_path, shots, cfg)
    elapsed = time.perf_counter() - t0

    total_kfs = sum(len(ks) for ks in result)
    print(f"Elapsed:       {elapsed:.2f} s")
    print(f"Keyframes:     {total_kfs}")
    print(f"Throughput:    {total_duration / elapsed:.1f} s-of-video / s")
    print(f"Per keyframe:  {(elapsed / total_kfs * 1000):.1f} ms" if total_kfs else "-")
    print()

    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        for shot_idx, shot_kfs in enumerate(result):
            for kf in shot_kfs:
                name = f"shot{shot_idx:04d}_bucket{kf.bucket_index:02d}_t{kf.timestamp_sec:08.2f}_sharp{kf.sharpness:08.1f}.jpg"
                kf.image.save(dump_dir / name, quality=92)
        print(f"Keyframes dumped to: {dump_dir}")


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("Usage: bench_real_video.py VIDEO_PATH [SHOTS_JSON] [--dump OUT_DIR]")
        return 2

    video_path = Path(argv[0]).expanduser().resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    shots_json: Path | None = None
    dump_dir: Path | None = None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--dump":
            dump_dir = Path(argv[i + 1]).expanduser().resolve()
            i += 2
        else:
            shots_json = Path(a).expanduser().resolve()
            i += 1

    shots = _load_shots(shots_json) if shots_json else _whole_video_as_one_shot(video_path)
    run(video_path, shots, dump_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
