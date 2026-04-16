# findit-keyframe

Per-shot keyframe extraction with stratified temporal sampling.

> **Status**: Python reference implementation. The Rust translation is the long-term target — see [`docs/rust-porting.md`](docs/rust-porting.md). Every module is written for a 1:1 port.

`findit-keyframe` consumes shot boundaries (e.g., from [`scenesdetect`](https://github.com/Findit-AI/scenesdetect)) and selects 1–N high-quality, **temporally distributed** frames per shot so downstream models — vision-language (Qwen3-VL-2B), embeddings (SigLIP 2), saliency (Apple Vision) — see the temporal progression of each shot, not just one representative moment.

## Why not "one frame per shot"

A single keyframe per shot loses temporal information. A 30-second talking-head shot and a 30-second action sequence look identical to a downstream VLM if you give it one frame. We split each shot into equal-duration **bins**, run a quality gate on candidates within each bin, and pick the best one — yielding a small, well-spaced sequence the VLM can reason about.

## Install

```bash
pip install -e ".[dev]"             # core
pip install -e ".[dev,macos]"        # + Apple Vision saliency provider
```

Requires Python ≥ 3.11. Core deps: `av` (PyAV) and `numpy`. No OpenCV.

## Quickstart

```python
from pathlib import Path
from findit_keyframe import (
    SamplingConfig, ShotRange, Timebase, Timestamp,
    VideoDecoder, default_saliency_provider, extract_all,
)

shots = [
    ShotRange(
        start=Timestamp(0,    Timebase(1, 1000)),
        end  =Timestamp(5000, Timebase(1, 1000)),
    ),
    # ... more shots from scenesdetect ...
]
decoder = VideoDecoder.open(Path("my_video.mp4"))
keyframes = extract_all(shots, decoder, SamplingConfig(), default_saliency_provider())
for shot_keyframes in keyframes:
    for kf in shot_keyframes:
        print(kf.timestamp.seconds, kf.confidence, kf.quality.laplacian_var)
```

## CLI

```bash
findit-keyframe extract video.mp4 shots.json out/ --saliency apple
```

The shot JSON schema and manifest output schema are documented in [`docs/algorithm.md`](docs/algorithm.md).

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT license ([`LICENSE-MIT`](LICENSE-MIT))

at your option.
