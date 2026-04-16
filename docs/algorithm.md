# Algorithm Specification

> **Audience**: Implementers (Python authors, Rust translators) and reviewers. Language-agnostic.

## 0. Overview

Given a video and a list of shots (half-open `[start, end)` time ranges from `scenesdetect`), produce, per shot, a small ordered list of **keyframes** that:

1. Are **temporally distributed** across the shot (no clustering at one timestamp).
2. Pass a **hard quality gate** (not black, not blown-out, not flat).
3. Maximise a **soft quality score** within their respective sub-windows.
4. Optionally weight a **saliency signal** (Apple Vision attention) when available.

A keyframe carries: source shot id, timestamp, originating bin index, decoded RGB pixels, quality metrics, and a `Confidence` tag (`High` / `Low` / `Degraded`).

## 1. Stratified Temporal Sampling

For a shot of duration `D` seconds, target interval `I` (default 4.0s), and per-shot cap `M` (default 16):

```
N = clamp(ceil(D / I), 1, M)
```

The shot is partitioned into `N` equal-duration **bins**. The first and last bin are shrunk inward by `boundary_shrink_pct` (default 2 %) of `D` to avoid sampling on cut transitions.

## 2. Within-Bin Selection

For each bin:

1. **Probe**: choose `K = candidates_per_bin` (default 6) candidate timestamps uniformly within the bin.
2. **Decode** each candidate frame (delegated to the decoder; strategy is its concern).
3. **Hard gate** each candidate via `QualityGate`:
   - `mean_luma ∈ [15/255, 240/255]`
   - `luma_variance ≥ 5`
4. **Score** survivors:
   ```
   score = 0.6 · norm(laplacian_var)
         + 0.2 · norm(entropy)
         + 0.2 · saliency_mass     # 0 when no saliency provider
   ```
   `norm` is percentile-rank within the bin's surviving pool.
5. **Pick** `argmax` → emit `ExtractedKeyframe(confidence=High)`.

## 3. Fallback

If a bin has zero survivors after step 3:

1. Expand the search window by `fallback_expand_pct` (default 20 %) into adjacent bins, re-probe, re-gate.
2. If still none, force-pick the highest-score candidate (gate ignored) → emit with `confidence=Degraded`.

A bin emitting via the expanded window (step 1) is tagged `Low`. This three-level confidence makes downstream filtering trivial.

## 4. Invariants

- Bin count `N ≥ 1` for any non-zero-duration shot.
- Bins are disjoint and union-cover the shot interior (after boundary shrink).
- Output keyframe count per shot ∈ `[1, M]`, monotonically increasing in timestamp.
- Every `ExtractedKeyframe.bucket_index` is in `[0, N)`.

## 5. JSON Schemas

### Input — shots

```json
{
  "shots": [
    {
      "id": 0,
      "start_pts": 0,
      "end_pts": 1000,
      "timebase_num": 1,
      "timebase_den": 1000
    }
  ]
}
```

### Output — manifest

```json
{
  "video": "path/to/video.mp4",
  "keyframes": [
    {
      "shot_id": 0,
      "bucket": 0,
      "file": "kf_000_000.jpg",
      "timestamp_sec": 1.234,
      "quality": {
        "laplacian_var": 215.4,
        "mean_luma": 0.41,
        "luma_variance": 1820.7,
        "entropy": 7.31,
        "saliency_mass": 0.62
      },
      "confidence": "high"
    }
  ]
}
```

## 6. Parameter Rationale

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `target_interval_sec` | 4.0 | One frame per ~4s gives a VLM enough temporal cadence to detect changes without saturating context. |
| `candidates_per_bin` | 6 | Empirically enough to find a sharp frame at 24–60 fps in any 2–10s bin without runaway decode cost. |
| `max_frames_per_shot` | 16 | Hard cap to bound per-shot cost; long static shots still compress to 16. |
| `boundary_shrink_pct` | 0.02 | Avoids the ±1-frame uncertainty around a cut. |
| `fallback_expand_pct` | 0.20 | Symmetric expand into both neighbours covers most under-exposed openings. |
| `target_size` | 384 | Sweet spot for SigLIP 2 / Qwen3-VL inputs; resize is part of decode to amortise cost. |

## 7. Known Limitations

- **VFR (variable frame rate)**: PTS handling is correct, but quality scores compare across temporally-uneven samples. Not a defect for our shot lengths.
- **Decoder ±1 frame jitter**: requested timestamp may resolve to nearest I/P frame. Documented; tolerated.
- **Shot-spanning duplicates**: identical anchor-shot keyframes across consecutive shots are not deduplicated. Out of scope for P1; see TASKS.md §7.
