"""Instrumented benchmark: separates decode vs algorithm time, dumps stats.

Emits:
    * Per-video summary: decode / algorithm / total time breakdown
    * Per-shot breakdown: # candidates, # buckets, # survivors, fallback rate
    * Sharpness distribution of selected keyframes
    * Extracted JPEGs for visual review
    * HTML gallery (per-shot thumbnails side by side)
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np
from PIL import Image

from findit_keyframe.decode import FrameCandidate, decode_shot_candidates
from findit_keyframe.quality import score_frame
from findit_keyframe.selector import compute_n_buckets
from findit_keyframe.types import Config, Keyframe, Shot

if TYPE_CHECKING:
    pass


class Timer:
    def __init__(self) -> None:
        self.decode_s = 0.0
        self.algo_s = 0.0
        self.total_s = 0.0

    @contextmanager
    def decode(self):
        t = time.perf_counter()
        yield
        self.decode_s += time.perf_counter() - t

    @contextmanager
    def algo(self):
        t = time.perf_counter()
        yield
        self.algo_s += time.perf_counter() - t


def _extract_instrumented(
    video_path: Path, shots: list[Shot], config: Config, timer: Timer
) -> tuple[list[list[Keyframe]], list[dict]]:
    """Run extract() but time decode and algorithm separately, collect per-shot stats."""
    results: list[list[Keyframe]] = []
    stats: list[dict] = []

    t_all = time.perf_counter()
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for shot_idx, shot in enumerate(shots):
            n_buckets = compute_n_buckets(shot, config)

            # Decode all candidates.
            with timer.decode():
                candidates = decode_shot_candidates(container, stream, shot, n_buckets, config)

            # Algorithm: score + select per bucket.
            with timer.algo():
                keyframes, shot_stats = _select_with_stats(
                    candidates, config, shot_idx, n_buckets
                )
            results.append(keyframes)
            stats.append(shot_stats)

    timer.total_s = time.perf_counter() - t_all
    return results, stats


def _select_with_stats(
    candidates: list[FrameCandidate],
    config: Config,
    shot_idx: int,
    n_buckets: int,
) -> tuple[list[Keyframe], dict]:
    """Selector with per-shot statistics — uses the production `score_frame`
    path so timings match what users actually pay."""
    by_bucket: dict[int, list[FrameCandidate]] = {}
    for c in candidates:
        by_bucket.setdefault(c.bucket_index, []).append(c)

    shot_stat = {
        "shot_idx": shot_idx,
        "n_buckets": n_buckets,
        "total_candidates": len(candidates),
        "empty_buckets": n_buckets - len(by_bucket),
        "fallback_buckets": 0,
        "sharpness_values": [],
        "picks": [],
    }

    picks: list[Keyframe] = []
    for b_idx in sorted(by_bucket.keys()):
        bucket = by_bucket[b_idx]

        # One score per candidate — zero redundant compute.
        scored = [(cand, score_frame(cand.rgb)) for cand in bucket]

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

        used_fallback = not strict
        if used_fallback:
            shot_stat["fallback_buckets"] += 1
        pool = strict if strict else scored
        best_cand, best_score = max(pool, key=lambda p: p[1].sharpness)

        img = Image.fromarray(best_cand.rgb, mode="RGB")
        kf = Keyframe(
            timestamp_sec=best_cand.timestamp_sec,
            image=img,
            sharpness=best_score.sharpness,
            brightness=best_score.brightness,
            bucket_index=b_idx,
        )
        picks.append(kf)
        shot_stat["sharpness_values"].append(best_score.sharpness)
        shot_stat["picks"].append(
            {
                "bucket": b_idx,
                "ts": best_cand.timestamp_sec,
                "sharp": best_score.sharpness,
                "brightness": best_score.brightness,
                "fallback": used_fallback,
            }
        )

    picks.sort(key=lambda k: k.timestamp_sec)
    return picks, shot_stat


# ---------- Reporting ---------------------------------------------------------


def _summarise(name: str, shots: list[Shot], results: list[list[Keyframe]], stats: list[dict], timer: Timer) -> dict:
    total_kfs = sum(len(ks) for ks in results)
    total_cands = sum(s["total_candidates"] for s in stats)
    total_buckets = sum(s["n_buckets"] for s in stats)
    empty_buckets = sum(s["empty_buckets"] for s in stats)
    fallback_buckets = sum(s["fallback_buckets"] for s in stats)

    all_sharps: list[float] = []
    for s in stats:
        all_sharps.extend(s["sharpness_values"])

    span = sum(s.duration_sec for s in shots)
    return {
        "name": name,
        "shots": len(shots),
        "span_sec": span,
        "total_buckets": total_buckets,
        "empty_buckets": empty_buckets,
        "fallback_buckets": fallback_buckets,
        "total_candidates": total_cands,
        "total_keyframes": total_kfs,
        "decode_s": timer.decode_s,
        "algo_s": timer.algo_s,
        "total_s": timer.total_s,
        "sharpness": {
            "min": float(np.min(all_sharps)) if all_sharps else 0.0,
            "p25": float(np.percentile(all_sharps, 25)) if all_sharps else 0.0,
            "median": float(np.median(all_sharps)) if all_sharps else 0.0,
            "p75": float(np.percentile(all_sharps, 75)) if all_sharps else 0.0,
            "max": float(np.max(all_sharps)) if all_sharps else 0.0,
        },
    }


def _print_summary(summary: dict) -> None:
    print(f"\n══════  {summary['name']}  ══════")
    print(f"Shots:            {summary['shots']}")
    print(f"Video span:       {summary['span_sec']:.1f} s")
    print(f"Buckets total:    {summary['total_buckets']}")
    print(f"Buckets empty:    {summary['empty_buckets']}")
    print(
        f"Buckets fallback: {summary['fallback_buckets']} "
        f"({summary['fallback_buckets']/max(summary['total_buckets'],1)*100:.1f}%)"
    )
    print(f"Candidates:       {summary['total_candidates']}")
    print(f"Keyframes:        {summary['total_keyframes']}")
    print()
    print(f"Time decode:      {summary['decode_s']:.2f} s")
    print(f"Time algorithm:   {summary['algo_s']:.3f} s   ← pure selection/quality compute")
    print(f"Time total:       {summary['total_s']:.2f} s")
    print(
        f"Algo per KF:      {summary['algo_s']/max(summary['total_keyframes'],1)*1000:.2f} ms"
    )
    print(f"Algo share:       {summary['algo_s']/max(summary['total_s'],1e-6)*100:.1f}% of total")
    print(f"Realtime factor:  {summary['span_sec']/max(summary['total_s'],1e-6):.1f}×")
    print()
    s = summary["sharpness"]
    print("Selected sharpness distribution:")
    print(f"  min / p25 / median / p75 / max = "
          f"{s['min']:.0f} / {s['p25']:.0f} / {s['median']:.0f} / {s['p75']:.0f} / {s['max']:.0f}")


def _dump_jpegs(results: list[list[Keyframe]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for shot_idx, kfs in enumerate(results):
        for kf in kfs:
            fb = "_fb" if kf.sharpness < Config().min_sharpness else ""
            name = f"s{shot_idx:04d}_b{kf.bucket_index:02d}_t{kf.timestamp_sec:08.2f}_sh{kf.sharpness:07.0f}{fb}.jpg"
            kf.image.save(out_dir / name, quality=88)


def _dump_html_gallery(
    video_name: str,
    shots: list[Shot],
    results: list[list[Keyframe]],
    stats: list[dict],
    summary: dict,
    out_dir: Path,
) -> Path:
    out = out_dir / f"{video_name}_gallery.html"
    rows: list[str] = []
    for shot_idx, (shot, kfs, stat) in enumerate(zip(shots, results, stats)):
        thumbs = "".join(
            f'<div class="kf"><img src="{video_name}/s{shot_idx:04d}_b{kf.bucket_index:02d}_t{kf.timestamp_sec:08.2f}_sh{kf.sharpness:07.0f}'
            f'{"_fb" if kf.sharpness < Config().min_sharpness else ""}.jpg" loading="lazy">'
            f'<div class="cap">b{kf.bucket_index} · {kf.timestamp_sec:.2f}s · '
            f'sh={kf.sharpness:.0f}{" ⚠fb" if kf.sharpness < Config().min_sharpness else ""}</div></div>'
            for kf in kfs
        )
        dur = shot.duration_sec
        fb_tag = (
            f'<span class="warn">{stat["fallback_buckets"]} fb</span>'
            if stat["fallback_buckets"]
            else ""
        )
        rows.append(
            f'<section><h3>Shot {shot_idx}: {shot.start_sec:.2f}–{shot.end_sec:.2f}s '
            f'(dur={dur:.2f}s, {stat["n_buckets"]} buckets, {len(kfs)} picks {fb_tag})</h3>'
            f'<div class="row">{thumbs}</div></section>'
        )

    s = summary["sharpness"]
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{video_name} — keyframes</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f7; }}
  h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 8px; }}
  section {{ margin: 20px 0; background: #fff; padding: 12px; border-radius: 8px; }}
  h3 {{ font-size: 14px; margin: 0 0 12px; color: #444; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .kf {{ max-width: 220px; }}
  .kf img {{ width: 100%; border-radius: 4px; display: block; }}
  .cap {{ font-size: 11px; color: #666; margin-top: 4px; font-family: monospace; }}
  .warn {{ color: #c33; font-weight: 600; }}
  table {{ border-collapse: collapse; font-size: 13px; }}
  td, th {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; }}
</style></head><body>
<h2>{video_name}</h2>
<table>
  <tr><th>Shots</th><td>{summary['shots']}</td>
      <th>Span</th><td>{summary['span_sec']:.1f} s</td>
      <th>Keyframes</th><td>{summary['total_keyframes']}</td></tr>
  <tr><th>Decode</th><td>{summary['decode_s']:.2f} s</td>
      <th>Algorithm</th><td>{summary['algo_s']:.3f} s</td>
      <th>Realtime</th><td>{summary['span_sec']/max(summary['total_s'],1e-6):.1f}×</td></tr>
  <tr><th>Fallback</th><td>{summary['fallback_buckets']}/{summary['total_buckets']}</td>
      <th>Sharpness median</th><td>{s['median']:.0f}</td>
      <th>Range</th><td>{s['min']:.0f} – {s['max']:.0f}</td></tr>
</table>
{''.join(rows)}
</body></html>"""
    out.write_text(html, encoding="utf-8")
    return out


# ---------- Main --------------------------------------------------------------


def run_one(name: str, video_path: Path, shots_json: Path, out_root: Path) -> dict:
    with shots_json.open() as f:
        data = json.load(f)
    shots = [Shot(start_sec=s["start_sec"], end_sec=s["end_sec"]) for s in data["shots"]]

    timer = Timer()
    results, stats = _extract_instrumented(video_path, shots, Config(), timer)
    summary = _summarise(name, shots, results, stats, timer)
    _print_summary(summary)

    jpeg_dir = out_root / name
    _dump_jpegs(results, jpeg_dir)
    html = _dump_html_gallery(name, shots, results, stats, summary, out_root)
    print(f"HTML gallery:     {html}")
    return summary


def main() -> int:
    out_root = Path.home() / "Downloads" / "keyframe-eval"
    out_root.mkdir(parents=True, exist_ok=True)

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

    summaries = []
    for name, video, shots in jobs:
        if not video.exists() or not shots.exists():
            print(f"SKIP {name}: {video if not video.exists() else shots} not found")
            continue
        summaries.append(run_one(name, video, shots, out_root))

    # Save combined JSON
    combined = out_root / "summary.json"
    combined.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"\nCombined summary: {combined}")
    print(f"\nOpen in browser:\n  file://{out_root}/kino_gallery.html")
    print(f"  file://{out_root}/kurutta_gallery.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
