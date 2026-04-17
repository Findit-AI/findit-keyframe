# findit-keyframe

Extract temporally-distributed, quality-aware keyframes from video shots —
tuned for downstream VLM consumption (SigLIP 2, Apple Vision, Qwen3-VL).

## What it does

**Input**: a video file and a list of shot time ranges (from upstream scene detector).
**Output**: for each shot, `N` PIL images chosen to cover the shot uniformly in time
and maximize per-frame sharpness.

```
Shot ──► [divide into N time buckets] ──► [sample candidates per bucket]
                                              │
                                              ▼
                                    [filter black/overexposed/solid-color]
                                              │
                                              ▼
                                    [score by Tenengrad sharpness]
                                              │
                                              ▼
                                      [pick argmax per bucket]
                                              │
                                              ▼
                                     N Keyframes (PIL.Image, sorted by time)
```

## Why this algorithm

| Naive approach | Problem |
|----------------|---------|
| Uniform sampling (every 2s) | Hits random frames → 20-30% are blurry / black / transition |
| Single representative frame (medoid) | Loses temporal information VLM needs |
| Deep-learning-based selection | Circular: uses VLM to pick frames for VLM |

**This library**: stratified time buckets guarantee temporal coverage, per-bucket
quality scoring guarantees sharpness. No ML dependencies.

## Install

```bash
uv pip install -e .        # from source
# or
pip install findit-keyframe
```

## Use

```python
from pathlib import Path
from findit_keyframe import extract, Shot, Config

# Shots typically come from upstream scene detection (e.g. scenesdetect)
shots = [
    Shot(start_sec=0.0,  end_sec=12.3),
    Shot(start_sec=12.3, end_sec=47.8),
]

keyframes_per_shot = extract(
    video_path=Path("input.mp4"),
    shots=shots,
    config=Config(target_interval_sec=4.0, max_frames_per_shot=16),
)

for shot_idx, keyframes in enumerate(keyframes_per_shot):
    for kf in keyframes:
        kf.image.save(f"shot{shot_idx}_{kf.timestamp_sec:.2f}.jpg")
        # kf.image is a PIL.Image ready for SigLIP / Qwen3-VL processors
```

## Config knobs

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `target_interval_sec` | 4.0 | Aim for one keyframe every N seconds |
| `max_frames_per_shot` | 16 | Hard cap (VLM token budget) |
| `candidates_per_bucket` | 6 | Candidates decoded per bucket, best one wins |
| `min_sharpness` | 50.0 | Tenengrad threshold (blurry frames rejected) |
| `black_mean_threshold` | 15.0 | Y-mean below → reject as black frame |
| `bright_mean_threshold` | 240.0 | Y-mean above → reject as overexposed |
| `variance_threshold` | 5.0 | Y-variance below → reject as solid-color |
| `margin_ratio` | 0.02 | Skip first/last 2% of shot (transition padding) |

## Contract with upstream scene detector

Just timestamps as `float` seconds. No shared types, no Rust bindings needed.

```json
{"shots": [{"start_sec": 0.0, "end_sec": 12.3}, ...]}
```

## License

MIT
