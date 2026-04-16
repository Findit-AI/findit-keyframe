# Algorithm Specification

> **Audience**: Implementers (Python authors, Rust translators) and reviewers. Language-agnostic.

## 0. Overview

Given a video and a list of shots (half-open `[start, end)` time ranges from `scenesdetect`), produce, per shot, a small ordered list of **keyframes** that:

1. Are **temporally distributed** across the shot (no clustering at one timestamp).
2. Pass a **hard quality gate** (not black, not blown-out, not flat).
3. Maximise a **soft quality score** within their respective sub-windows.
4. Optionally weight a **saliency signal** (Apple Vision attention) when available.

A keyframe carries: source shot id, timestamp (the actual decoded PTS), originating bin index, decoded RGB pixels, quality metrics, and a `Confidence` tag (`High` / `Low` / `Degraded`).

## 1. Stratified Temporal Sampling

For a shot of duration `D` seconds, target interval `I` (default 4.0 s), and per-shot cap `M` (default 16):

```
N = clamp(ceil(D / I), 1, M)
```

The shot is partitioned into `N` equal-duration **bins** over the symmetrically shrunken effective range `[start + s, end - s]` where `s = boundary_shrink_pct * D` (default 2 %). Shrinking the outer edges keeps the algorithm away from the ±1-frame uncertainty around scenesdetect cuts.

## 2. Within-Bin Selection

For each bin `[t0, t1)`:

1. **Probe** `K = candidates_per_bin` (default 6) candidate timestamps at the **cell centres** of the bin: `t0 + (i + 0.5) * (t1 - t0) / K` for `i ∈ [0, K)`. Cell centres avoid sampling exactly on bin edges, where the upstream cut detector is least confident.
2. **Decode** each candidate frame through the decoder (strategy is its concern).
3. **Hard gate** each candidate via `QualityGate`:
   - `mean_luma ∈ [15/255, 240/255]` (inclusive)
   - `luma_variance ≥ 5` (sample variance, ddof = 1, on the raw 0–255 scale)
4. **Score** survivors with the composite:
   ```
   score = 0.6 * rank(laplacian_var)
         + 0.2 * rank(entropy)
         + 0.2 * saliency_mass        # 0.0 when no saliency provider is wired
   ```
   `rank` is **stable ordinal rank** within the bin's surviving pool, scaled to `[0, 1]`. A single-survivor bin gets `rank = 1` for both metrics, so its score collapses to `0.8 + 0.2 * saliency_mass`.
5. **Pick** `argmax` → emit `ExtractedKeyframe(confidence=High)`.

## 3. Fallback

If a bin has zero survivors after step 2.3:

1. **Expand** the search window symmetrically by `fallback_expand_pct * (t1 - t0)` (default 20 % of bin width) on each side, clamped to the shot's `[start.seconds, end.seconds]`. Re-probe `K` candidates in the expanded window, decode, gate, score. A surviving pick → `Confidence.Low`.
2. **Force-pick**: if the expanded window also yields no survivors, gather the union of the native-bin and expanded-bin candidates, score them all (gate ignored), and pick `argmax`. Emit `Confidence.Degraded`.

The three-level confidence makes downstream filtering trivial: drop `Degraded` if you only want curated frames, or accept everything if you'd rather get something for every bin.

## 4. Saliency Provider Contract

A `SaliencyProvider` exposes a single method `compute(rgb: ndarray) -> float` returning a saliency mass in `[0.0, 1.0]`. Two implementations ship:

- `NoopSaliencyProvider` — always returns `0.0`. Default everywhere.
- `AppleVisionSaliencyProvider` — wraps `VNGenerateAttentionBasedSaliencyImageRequest`. The scalar is `clamp(sum(area * confidence), 0, 1)` over the request's `salientObjects` bounding boxes; the heatmap `CVPixelBuffer` is intentionally *not* read because the bounding-box scalar carries the same "is anything attention-grabbing" signal with a much cleaner pyobjc API. The Rust port (`objc2-vision`) may pick either path.

`default_saliency_provider()` returns Apple Vision on macOS when `pyobjc-framework-Vision` is installed, else `Noop`.

## 5. Invariants

- Bin count `N ≥ 1` for any non-zero-duration shot.
- Bins are disjoint and union-cover the (post-shrink) shot interior.
- Output keyframe count per shot is exactly `N`, monotonically non-decreasing in `bucket_index`.
- Every `ExtractedKeyframe.bucket_index` is in `[0, N)`.
- Saliency mass is in `[0.0, 1.0]`; quality fields are real-valued and finite.

## 6. JSON Schemas

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

`timestamp_sec` is the actual decoded PTS, which may differ from the probe time by ±1 frame because the seek lands on the nearest preceding keyframe and decoding stops at the first frame whose PTS is at or after the target.

## 7. Parameter Rationale

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `target_interval_sec` | 4.0 | One frame per ~4 s gives a VLM enough temporal cadence to detect changes without saturating context. |
| `candidates_per_bin` | 6 | Empirically enough to find a sharp frame at 24-60 fps in any 2-10 s bin without runaway decode cost. |
| `max_frames_per_shot` | 16 | Hard cap to bound per-shot cost; long static shots still compress to 16. |
| `boundary_shrink_pct` | 0.02 | Avoids the ±1-frame uncertainty around a cut. |
| `fallback_expand_pct` | 0.20 | Symmetric expand into both neighbours covers most under-exposed openings. |
| `target_size` | 384 | Sweet spot for SigLIP 2 / Qwen3-VL inputs; resize is part of decode to amortise cost. |
| Quality gate `mean_luma` | `[15/255, 240/255]` | Reject black-frame openings and blown-out flashes; range matches BT.601 limited-range Y / 255. |
| Quality gate `luma_variance` | `≥ 5` | Reject pixel-flat frames (e.g. a single solid colour). |

## 8. Known Limitations

- **VFR (variable frame rate)**: PTS handling is correct, but quality scores compare across temporally-uneven samples. Not a defect for our shot lengths.
- **Decoder ±1 frame jitter**: requested timestamp may resolve to nearest I/P frame.
- **Shot-spanning duplicates**: identical anchor-shot keyframes across consecutive shots are not deduplicated. `TASKS.md` §7 ("Out-of-Scope but Noted for Later") tracks the MMR cross-bin deduplication path alongside four other deferred items (SigLIP medoid selection, cross-shot coherence, learned quality models, hardware decode); all are tagged P2+ for the Rust phase.
- **Sequential decode strategy**: `pick_strategy` returns `Sequential` for dense shot lists, but the current implementation always uses `PerShotSeek`. The Sequential optimisation lands in P3+ (Rust phase).
