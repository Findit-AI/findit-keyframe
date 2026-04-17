"""End-to-end example: extract keyframes given a shots JSON from upstream.

This mirrors how the FindIt indexer will call the library.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from findit_keyframe import Config, Shot, extract


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: extract_from_shots.py VIDEO SHOTS_JSON OUTPUT_DIR")
        return 2

    video = Path(sys.argv[1])
    shots_json = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(shots_json.read_text())
    shots = [Shot(start_sec=s["start_sec"], end_sec=s["end_sec"]) for s in data["shots"]]

    keyframes_per_shot = extract(video, shots, Config())

    for shot_idx, kfs in enumerate(keyframes_per_shot):
        for kf in kfs:
            name = f"shot{shot_idx:04d}_t{kf.timestamp_sec:08.2f}.jpg"
            kf.image.save(output_dir / name, quality=92)

    total = sum(len(ks) for ks in keyframes_per_shot)
    print(f"Wrote {total} keyframes across {len(shots)} shots to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
