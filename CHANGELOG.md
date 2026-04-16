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
