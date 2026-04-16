# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Repo scaffolding: `pyproject.toml` with hatchling build, dual Apache-2.0/MIT license, ruff + mypy strict configs, pytest setup.
- `src/findit_keyframe/` module skeleton: `types`, `decoder`, `quality`, `sampler`, `saliency`, `cli`.
- `docs/algorithm.md` and `docs/rust-porting.md` skeletons for algorithm spec and Rust translation map.
- GitHub Actions CI: ruff, mypy, pytest on push/PR for Python 3.11 and 3.12.
- Quality module: `rgb_to_luma` (BT.601 fixed-point), `laplacian_variance`, `mean_luma`, `luma_variance`, `entropy`, `QualityGate`, `compute_quality`.
- Decoder module: `VideoDecoder` (PyAV backend, context-manager) with `decode_at` (keyframe seek + forward decode) and `decode_sequential` (linear pass over a shot list); `Strategy` enum and `pick_strategy` density heuristic.
- Sampler module: `compute_bins` (boundary-shrunken equal partition), `score_bin_candidates` (ordinal-rank composite), `select_from_bin`, fallback path with `Confidence.Low` / `Confidence.Degraded`, top-level `extract_for_shot` and `extract_all`.
- CLI (`findit-keyframe extract VIDEO SHOTS_JSON OUTPUT`): scenesdetect-compatible shot JSON parsing, optional `--config` for `SamplingConfig` overrides, `--saliency {none,apple}` flag (apple stub returns input-error until T6), per-keyframe baseline JPEG output via PyAV's mjpeg / image2 muxer, `manifest.json` output with quality dict and confidence string. Exit codes: 0 success, 1 input error, 2 extraction error.
- Benchmark script (`benchmarks/bench_e2e.py`): standalone CLI, optional shot JSON, configurable `--target-size`, append-only `results.md` log with date and git SHA, peak-memory normalised across Linux/macOS.
- Saliency providers (`saliency.py`): `SaliencyProvider` runtime-checkable Protocol, `NoopSaliencyProvider` (always 0.0), `AppleVisionSaliencyProvider` (macOS, wraps `VNGenerateAttentionBasedSaliencyImageRequest` via pyobjc; lazy import so non-macOS imports don't crash), `default_saliency_provider()` factory.
- Sampler now accepts an optional `saliency_provider` argument on `select_from_bin`, `extract_for_shot`, and `extract_all`; saliency mass feeds the composite score weight.
- CLI `--saliency apple` now wires `AppleVisionSaliencyProvider` end-to-end (was a stub returning input-error in P3).
- Documentation finalised: `docs/algorithm.md` reflects shipped behaviour (cell-centred sampling, ordinal-rank scoring, three-tier fallback) and adds a Saliency Provider Contract section. `docs/rust-porting.md` carries a complete type map (including `DecodedFrame`, `Strategy`, `QualityGate`, all three `SaliencyProvider` impls), the actual numpy ops used, and an Apple Vision idiom map for `objc2-vision`.
