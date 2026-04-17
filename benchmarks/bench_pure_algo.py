"""Strict decode-vs-algorithm separation.

Step 1: decode ALL candidate frames into memory (fully ffmpeg-bound)
Step 2: run ONLY the algorithm (quality scoring + selection) on pre-decoded
        numpy arrays, time it separately.

Step 2 is bit-for-bit what ``findit_keyframe.selector.select_keyframes`` does,
executed outside any ffmpeg I/O so we can report a clean pure-algorithm
number.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import av
from PIL import Image

from findit_keyframe.decode import FrameCandidate, decode_shot_candidates
from findit_keyframe.quality import score_frame
from findit_keyframe.selector import compute_n_buckets
from findit_keyframe.types import Config, Keyframe, Shot


def decode_all_candidates(
    video_path: Path, shots: list[Shot], config: Config
) -> tuple[list[list[FrameCandidate]], float]:
    """Step 1: decode everything. Returns candidates per shot + ffmpeg time."""
    all_candidates: list[list[FrameCandidate]] = []
    t0 = time.perf_counter()
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for shot in shots:
            n_buckets = compute_n_buckets(shot, config)
            cands = decode_shot_candidates(container, stream, shot, n_buckets, config)
            all_candidates.append(cands)
    ffmpeg_time = time.perf_counter() - t0
    return all_candidates, ffmpeg_time


def run_algorithm_only(
    all_candidates: list[list[FrameCandidate]], config: Config
) -> tuple[list[list[Keyframe]], float, dict]:
    """Step 2: quality + selection only, using the production ``score_frame`` path.

    This is exactly what ``selector._select_from_bucket`` does — we inline it
    here purely to attach per-stage timers.
    """
    results: list[list[Keyframe]] = []

    t_score = 0.0  # score_frame (downscale + cvtColor + Sobel + stats in one pass)
    t_filter = 0.0  # FrameScore.is_unusable + sharpness floor check
    t_select = 0.0  # max() and dict bucketing
    t_pil = 0.0  # Image.fromarray on the chosen candidate

    t_total = time.perf_counter()

    for shot_candidates in all_candidates:
        # Bucket assignment (dict only, trivial work)
        t = time.perf_counter()
        by_bucket: dict[int, list[FrameCandidate]] = {}
        for c in shot_candidates:
            by_bucket.setdefault(c.bucket_index, []).append(c)
        t_select += time.perf_counter() - t

        picks: list[Keyframe] = []
        for b_idx in sorted(by_bucket.keys()):
            bucket = by_bucket[b_idx]

            # Score every candidate once.
            t = time.perf_counter()
            scored = [(cand, score_frame(cand.rgb)) for cand in bucket]
            t_score += time.perf_counter() - t

            # Filter + pool selection.
            t = time.perf_counter()
            strict = [
                (c, s)
                for c, s in scored
                if not s.is_unusable(
                    config.black_mean_threshold,
                    config.bright_mean_threshold,
                    config.luma_variance_threshold,
                    config.sat_variance_threshold,
                    config.max_clipping,
                )
                and s.sharpness >= config.min_sharpness
            ]
            t_filter += time.perf_counter() - t

            t = time.perf_counter()
            pool = strict if strict else scored
            best_cand, best_score = max(pool, key=lambda p: p[1].sharpness)
            t_select += time.perf_counter() - t

            t = time.perf_counter()
            img = Image.fromarray(best_cand.rgb, mode="RGB")
            t_pil += time.perf_counter() - t

            picks.append(
                Keyframe(
                    timestamp_sec=best_cand.timestamp_sec,
                    image=img,
                    sharpness=best_score.sharpness,
                    brightness=best_score.brightness,
                    bucket_index=b_idx,
                )
            )

        picks.sort(key=lambda k: k.timestamp_sec)
        results.append(picks)

    total_algo = time.perf_counter() - t_total
    breakdown = {
        "score_frame_s": t_score,
        "filter_s": t_filter,
        "select_s": t_select,
        "pil_convert_s": t_pil,
        "other_s": total_algo - (t_score + t_filter + t_select + t_pil),
    }
    return results, total_algo, breakdown


def run_one(name: str, video: Path, shots_json: Path) -> dict:
    with shots_json.open() as f:
        data = json.load(f)
    shots = [Shot(start_sec=s["start_sec"], end_sec=s["end_sec"]) for s in data["shots"]]
    config = Config()

    print(f"\n═══════════════  {name}  ═══════════════")
    print(f"Video: {video.name}")
    print(f"Shots: {len(shots)}  |  Span: {sum(s.duration_sec for s in shots):.1f} s")

    all_cands, t_ffmpeg = decode_all_candidates(video, shots, config)
    n_cands = sum(len(c) for c in all_cands)
    ram_mb = (
        sum(sum(c.rgb.nbytes for c in cands) for cands in all_cands) / 1024 / 1024
    )
    print(f"\nStep 1 (ffmpeg only):")
    print(f"  Candidates:         {n_cands}")
    print(f"  RGB RAM:            {ram_mb:.1f} MB")
    print(f"  Time:               {t_ffmpeg * 1000:.0f} ms")

    results, t_algo, breakdown = run_algorithm_only(all_cands, config)
    n_kfs = sum(len(r) for r in results)
    print(f"\nStep 2 (algorithm only, zero ffmpeg):")
    print(f"  Keyframes:          {n_kfs}")
    print(f"  Total algorithm:    {t_algo * 1000:.1f} ms")
    print(f"  Per candidate:      {t_algo / max(n_cands, 1) * 1000:.3f} ms")
    print(f"  Per keyframe:       {t_algo / max(n_kfs, 1) * 1000:.2f} ms")
    print(f"  Breakdown:")
    for k, v in breakdown.items():
        pct = v / t_algo * 100 if t_algo > 0 else 0
        print(f"    {k:18s}: {v * 1000:7.1f} ms  ({pct:5.1f}%)")

    print(f"\nSummary:")
    total = t_ffmpeg + t_algo
    print(f"  ffmpeg:             {t_ffmpeg * 1000:>8.0f} ms  ({t_ffmpeg / total * 100:5.1f}%)")
    print(f"  algorithm:          {t_algo * 1000:>8.0f} ms  ({t_algo / total * 100:5.1f}%)")
    print(f"  total:              {total * 1000:>8.0f} ms")

    return {
        "name": name,
        "shots": len(shots),
        "candidates": n_cands,
        "keyframes": n_kfs,
        "ffmpeg_ms": t_ffmpeg * 1000,
        "algo_ms": t_algo * 1000,
        "algo_per_cand_ms": t_algo / max(n_cands, 1) * 1000,
        "algo_per_kf_ms": t_algo / max(n_kfs, 1) * 1000,
        "breakdown_ms": {k: v * 1000 for k, v in breakdown.items()},
    }


def main() -> int:
    jobs = [
        (
            "kino",
            Path("/Users/cheongzhiyan/Downloads/Kino Demo Render.mp4"),
            Path("/tmp/kino_shots.json"),
        ),
        (
            "kurutta",
            Path("/Users/cheongzhiyan/Movies/POMeditMASTER/狂った一頁　編集済み  20241204.mp4"),
            Path("/tmp/kurutta_shots.json"),
        ),
    ]
    results = []
    for name, video, shots in jobs:
        if not video.exists() or not shots.exists():
            print(f"SKIP {name}")
            continue
        results.append(run_one(name, video, shots))

    out = Path.home() / "Downloads" / "keyframe-eval" / "pure_algo_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nJSON summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
