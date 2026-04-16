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
