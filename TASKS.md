# findit-keyframe — Task Document

**Target repo**: `github.com/Findit-AI/findit-keyframe` (to be created)
**Language**: Python (reference implementation), later translated to Rust by teammate
**Status**: Planning phase — this document is the source of truth for scope and acceptance

---

## 0. Context Recap

- **Upstream**: `scenesdetect` (Rust, published at `github.com/Findit-AI/scenesdetect` branch `0.1.0`) emits shot boundaries as `TimeRange` (start/end timestamps with timebase).
- **Downstream**: SigLIP 2 (vision embedding), Apple Vision (saliency/tags), Qwen3-VL-2B (VLM description).
- **Problem**: For each shot, select 1–N high-quality frames that are **temporally distributed** across the shot so VLMs can understand temporal progression, not just a single representative moment.
- **Architectural decision**: `findit-keyframe` is a **standalone repo**, Python reference first, then translated to Rust.
- **Why Python first**: Faster algorithm iteration, easier for teammate review before commitment to Rust.

---

## 1. Non-Goals (scope control)

To prevent scope creep, we explicitly exclude:

- ❌ Scene boundary detection (that's `scenesdetect`'s job; we consume its output).
- ❌ Training any ML selector (no labeled data, no budget).
- ❌ Video-level summarization (we're per-shot, not whole-video).
- ❌ GPU / hardware-accelerated decode in Python (Rust phase will add VideoToolbox).
- ❌ Distributed processing / cluster execution.
- ❌ Direct downstream model inference (SigLIP/Qwen). We only provide clean keyframe outputs; consumers run inference.
- ❌ Cross-shot deduplication (future optimization, not P1).

---

## 2. Project-Level Goals

| # | Goal | Why it matters |
|---|------|----------------|
| G1 | Produce a clean, reviewable Python reference implementation | Teammate can review algorithm correctness before committing to Rust |
| G2 | Every Python module has a 1:1 Rust translation path | Avoid Python-idiom traps (metaclasses, duck typing, dynamic dispatch) |
| G3 | Algorithmic correctness verifiable by deterministic fixtures | Rust translation can replay the same fixtures and match bit-for-bit (or near-bit) |
| G4 | Standalone repo, zero dependency on FindIt internals | Can be open-sourced; teammate clones independently |
| G5 | Reasonable Python performance (not a toy) | Must survive real videos (Kino Demo, Kurutta Ippeiji) for benchmarking |

---

## 3. Task Breakdown

Each task has:
- **Goal**: what the deliverable is
- **Scope**: what's included / excluded
- **Verification**: how we confirm it's done correctly

---

### Task 1 — Repo scaffolding

**Goal**: Create `findit-keyframe` repo with complete project structure, toolchain config, CI, and documentation skeleton.

**Scope (included)**:
- `pyproject.toml` with `hatchling` build backend, locked Python ≥ 3.11
- Dependencies: `av` (PyAV), `numpy`; optional `macos` extra for `pyobjc-framework-Vision`; dev extras for `pytest`, `ruff`, `mypy`
- Directory layout:
  ```
  src/findit_keyframe/{__init__.py, types.py, decoder.py, quality.py, sampler.py, saliency.py, cli.py}
  tests/
  benchmarks/
  examples/
  docs/
  ```
- `README.md`, `LICENSE` (Apache-2.0 + MIT dual to match scenesdetect), `CHANGELOG.md`, `.gitignore`
- `.github/workflows/ci.yml`: pytest, ruff, mypy on push/PR
- `ruff.toml` with strict lint rules matching the project ethos (line-length 100, import sorting, etc.)
- `mypy` in strict mode

**Scope (excluded)**:
- No actual algorithm code yet
- No fixture videos committed
- No benchmark numbers

**Verification**:
- [ ] `git clone` works on a fresh machine
- [ ] `pip install -e ".[dev,macos]"` succeeds on macOS
- [ ] `pytest` runs (0 tests collected is fine)
- [ ] `ruff check .` passes
- [ ] `mypy src/` passes (no code yet → trivially passes)
- [ ] CI green on first push
- [ ] `README.md` clearly states: "Python reference implementation; Rust translation target. See `docs/rust-porting.md`."

**Estimated effort**: 0.5–1 day

---

### Task 2 — Type definitions (`types.py`)

**Goal**: Mirror `scenesdetect`'s public types in Python using `@dataclass(frozen=True)` + explicit type hints, so Rust translation is 1:1.

**Scope (included)**:
- `Timebase(num: int, den: int)` — rational timebase, `den > 0` invariant
- `Timestamp(pts: int, timebase: Timebase)` — with `seconds` property
- `ShotRange(start: Timestamp, end: Timestamp)` — with `duration_sec` property
- `SamplingConfig` — user-tunable parameters with defaults:
  - `target_interval_sec: float = 4.0`
  - `candidates_per_bin: int = 6`
  - `max_frames_per_shot: int = 16`
  - `boundary_shrink_pct: float = 0.02`
  - `fallback_expand_pct: float = 0.20`
  - `target_size: int = 384`
- `Confidence` enum (`High`, `Low`, `Degraded`)
- `QualityMetrics(laplacian_var, mean_luma, luma_variance, entropy, saliency_mass)`
- `ExtractedKeyframe(shot_id, timestamp, bucket_index, rgb: bytes, width, height, quality, confidence)`

**Scope (excluded)**:
- No `__slots__` (interferes with dataclass in some versions; Rust doesn't need it anyway)
- No custom `__hash__` / `__eq__` unless required by algorithm (let `@dataclass(frozen=True)` generate them)
- No `Protocol` / ABC for decoder (we'll use a concrete class later; teammate can refactor)

**Verification**:
- [ ] `mypy --strict src/findit_keyframe/types.py` passes
- [ ] All dataclasses are `frozen=True` where immutable (everything except `SamplingConfig` and `ExtractedKeyframe`)
- [ ] Unit tests confirming:
  - `Timebase(0, 1)` raises (zero numerator is fine — scenesdetect allows it; zero denominator must fail)
  - `Timestamp.seconds` returns correct float for common cases (`1000 @ 1/1000 == 1.0s`)
  - `ShotRange.duration_sec` handles start/end in the same timebase correctly
- [ ] `docs/rust-porting.md` has a **Type Map** section showing Python ↔ Rust field correspondence for every type

**Estimated effort**: 0.5 day

---

### Task 3 — Video decoder (`decoder.py`)

**Goal**: PyAV-based frame decoder with two strategies (Sequential demux vs per-shot seek), auto-selected by shot density.

**Scope (included)**:
- `VideoDecoder` class:
  - `open(path: Path) -> VideoDecoder` — opens a video, caches metadata (fps, duration, time_base)
  - `decode_at(time_sec: float) -> DecodedFrame` — seek + decode one frame at/near the given time, returns RGB frame + PTS
  - `decode_sequential(shots: list[ShotRange]) -> Iterator[(shot_id, DecodedFrame)]` — single pass through the file, emits frames falling inside any shot's range
- Auto-strategy picker: `pick_strategy(shots, duration_sec) -> Strategy` based on density heuristic (density > 0.3 shots/sec OR shots > 200 → Sequential, else PerShotSeek)
- RGB conversion: decode in native format, convert to RGB24 at `target_size × target_size` via PyAV's built-in reformatter (wraps swscale)
- Frame metadata: PTS in video time_base, exposed as `Timestamp`

**Scope (excluded)**:
- No VideoToolbox hardware decode (macOS hwaccel belongs in Rust phase)
- No variable frame rate edge cases beyond basic PTS handling (document as known limitation)
- No audio decode

**Verification**:
- [ ] Unit test: decode a 10s synthetic video (generated via ffmpeg with known frame patterns), verify `decode_at(5.0)` returns the expected frame
- [ ] Integration test: open `Kino Demo Render.mp4`, decode first frame, verify dimensions match expected resolution (1920×1080)
- [ ] Performance sanity: 1-minute 1080p video, 100 `decode_at` calls should complete in < 5 seconds on M-series Mac (with PerShotSeek)
- [ ] Sequential strategy: decode a full 10s video (30 fps, so 300 frames), should yield 300 frames in order
- [ ] Strategy auto-selection: unit test with mock shots list, confirm density threshold picks correct strategy

**Estimated effort**: 1.5 days

---

### Task 4 — Quality metrics (`quality.py`)

**Goal**: Pure numpy implementation of per-frame quality signals. No OpenCV. Every function translates 1:1 to Rust `ndarray` or manual loops.

**Scope (included)**:
- `rgb_to_luma(rgb: np.ndarray) -> np.ndarray` — BT.601 integer: `(66*R + 129*G + 25*B + 128) >> 8 + 16`, matches scenesdetect's fixed-point choice
- `laplacian_variance(luma: np.ndarray) -> float` — 3×3 Laplacian kernel `[[0,1,0],[1,-4,1],[0,1,0]]`, return variance of the filtered output. Explicit loop + numpy, not `scipy.signal` or `cv2`.
- `mean_luma(luma: np.ndarray) -> float` — arithmetic mean, normalized to `[0.0, 1.0]`
- `luma_variance(luma: np.ndarray) -> float` — sample variance
- `entropy(luma: np.ndarray, bins: int = 256) -> float` — Shannon entropy of the 256-bin histogram, base-2 log. Manual numpy implementation (no scipy).
- `QualityGate` class with defaults matching our earlier decision:
  - Reject if `mean_luma` outside `[15/255, 240/255]`
  - Reject if `luma_variance` < 5
- `compute_quality(rgb: np.ndarray, saliency: Optional[float]) -> QualityMetrics` — composite, returns populated `QualityMetrics` dataclass

**Scope (excluded)**:
- No OpenCV, no scipy, no scikit-image
- No learned quality models (NIMA, MUSIQ)
- No motion blur detection beyond what Laplacian variance catches
- No Apple Vision integration in this module (that's `saliency.py`)

**Verification**:
- [ ] Unit test: synthetic all-black frame → `mean_luma ≈ 0.0`, `laplacian_var ≈ 0.0`, gate rejects
- [ ] Unit test: synthetic random noise frame → high laplacian_var, gate accepts
- [ ] Unit test: gradient image (smooth ramp) → low laplacian_var, high luma_variance, gate accepts
- [ ] Golden fixture test: pre-computed quality metrics JSON for 10 canonical frames (stored in `tests/fixtures/quality/`); Python output must match to 6 decimal places
- [ ] Performance: `compute_quality` on 384×384 RGB should complete in < 5 ms on M-series Mac
- [ ] `docs/rust-porting.md` shows the exact numpy op → ndarray op mapping for each function

**Estimated effort**: 1 day

---

### Task 5 — Sampler (`sampler.py`) — **core algorithm**

**Goal**: Implement stratified temporal sampling + within-bucket quality selection with graceful degradation. This is the algorithmic heart.

**Scope (included)**:
- `compute_bins(shot: ShotRange, config: SamplingConfig) -> list[tuple[float, float]]` — partition shot into N equal-duration bins with boundary shrinkage on first/last
- `score_candidate(quality: QualityMetrics) -> float` — weighted composite:
  ```
  score = 0.6 * normalized(laplacian_var)
        + 0.2 * normalized(entropy)
        + 0.2 * (saliency_mass or 0.0)
  ```
  where `normalized` is z-score or percentile-rank within the bin's candidate pool (document which)
- `select_from_bin(candidates: list[DecodedFrame], config: SamplingConfig) -> Optional[(DecodedFrame, QualityMetrics, Confidence)]` — apply quality gate, score, pick `argmax`; return `None` if all filtered out
- `fallback_pick(shot, bins, bin_idx, decoder, config)` — expand search to ±`fallback_expand_pct` of adjacent bins; if still nothing, force-pick highest-quality candidate with `Confidence.Degraded`
- `extract_for_shot(shot: ShotRange, decoder: VideoDecoder, config: SamplingConfig) -> list[ExtractedKeyframe]` — main entry point
- `extract_all(shots: list[ShotRange], decoder: VideoDecoder, config: SamplingConfig) -> list[list[ExtractedKeyframe]]` — process all shots; uses decoder's auto-strategy

**Scope (excluded)**:
- No MMR-based cross-bin deduplication (keep for P3)
- No CLIP/SigLIP-based relevance scoring (keep for P3)
- No cross-shot coherence
- No caching of decoded frames across shots (memory concerns; re-decode is cheap for sparse seek)

**Verification**:
- [ ] Unit test: synthetic 20s shot with known quality gradient (frames 0–100 sharp, 100–200 blurred, 200–300 sharp) — verify selected frames come from sharp regions in each bucket
- [ ] Unit test: N = `ceil(duration / 4.0)` with duration = 5s → N = 2; duration = 60s → N = 15; duration = 120s → N = 16 (capped)
- [ ] Unit test: black-frame shot — all candidates in first bucket fail hard gate → fallback picks degraded frame, `Confidence.Degraded`
- [ ] Unit test: bin boundaries — first bucket's `t0` is shifted by `boundary_shrink_pct`, last bucket's `t1` is shifted similarly
- [ ] Integration test: run on `Kino Demo Render.mp4` with scene cuts from scenesdetect, visually inspect top 5 shots — no blurry frames, no black frames, temporally distributed
- [ ] Regression fixture: JSON snapshot of `(shot_id, bin_index, timestamp, quality_score)` tuples for Kino Demo, deterministic across runs
- [ ] `docs/rust-porting.md` documents the exact algorithm in pseudocode (same as Python, but language-agnostic)

**Estimated effort**: 2 days

---

### Task 6 — Saliency adapter (`saliency.py`)

**Goal**: Optional Apple Vision saliency integration, with a no-op stub for non-macOS / Rust-future environments.

**Scope (included)**:
- `SaliencyProvider` protocol/interface: `compute(rgb: np.ndarray) -> float` (returns saliency mass in `[0, 1]`)
- `NoopSaliencyProvider` — always returns 0.0; used as default and on non-macOS
- `AppleVisionSaliencyProvider` (macOS only, guarded by `platform.system() == "Darwin"`) — uses `pyobjc-framework-Vision` to call `VNGenerateAttentionBasedSaliencyImageRequest`, sums the heatmap pixels
- Factory: `default_saliency_provider() -> SaliencyProvider` — returns Apple provider on macOS, Noop elsewhere

**Scope (excluded)**:
- No other saliency models (DeepGaze, UNISAL, etc.)
- No batch processing (one frame at a time; Rust can batch later)

**Verification**:
- [ ] Unit test: `NoopSaliencyProvider` always returns 0.0
- [ ] Integration test (macOS only, skipped elsewhere): `AppleVisionSaliencyProvider` on a known image (center-white, corners-black) returns a saliency value > 0.3
- [ ] Gracefully degrades: if `pyobjc-framework-Vision` is not installed, import of `AppleVisionSaliencyProvider` must not crash `findit_keyframe` package import
- [ ] `docs/rust-porting.md` shows how `AppleVisionSaliencyProvider` maps to `objc2-vision` + `VNGenerateAttentionBasedSaliencyImageRequest` in Rust

**Estimated effort**: 1 day (macOS testing adds overhead)

---

### Task 7 — CLI tool (`cli.py`)

**Goal**: Command-line utility for end-to-end testing and teammate demonstration.

**Scope (included)**:
- `findit-keyframe extract VIDEO_PATH SHOTS_JSON OUTPUT_DIR` — reads shots from JSON (scenesdetect-compatible format), extracts keyframes, writes them as JPEG files + a manifest JSON
- `--config CONFIG_JSON` — override `SamplingConfig` defaults
- `--saliency {none, apple}` — pick saliency provider
- Shot JSON schema (input):
  ```json
  {"shots": [{"id": 0, "start_pts": 0, "end_pts": 1000, "timebase_num": 1, "timebase_den": 1000}, ...]}
  ```
- Manifest JSON schema (output):
  ```json
  {"video": "path", "keyframes": [{"shot_id": 0, "bucket": 0, "file": "kf_000_000.jpg", "timestamp_sec": 1.2, "quality": {...}, "confidence": "high"}, ...]}
  ```
- Uses `argparse` (stdlib, no click/typer — cleaner Rust translation)

**Scope (excluded)**:
- No interactive TUI
- No progress bar via tqdm (keep deps minimal; log lines are fine)
- No direct scenesdetect invocation (user runs scenesdetect separately, pipes its JSON)

**Verification**:
- [ ] CLI help works: `findit-keyframe --help`
- [ ] End-to-end test: given a fixture video + shot JSON, produces expected number of JPEG files
- [ ] Manifest JSON is valid and round-trips through `json.loads`
- [ ] Exit codes: 0 on success, 1 on input errors, 2 on extraction failures

**Estimated effort**: 1 day

---

### Task 8 — Benchmarks (`benchmarks/`)

**Goal**: Establish a performance baseline that the future Rust translation must beat (target: 5–10× speedup).

**Scope (included)**:
- `bench_e2e.py` — run extraction on `Kino Demo Render.mp4` and `狂った一頁 編集済み.mp4`, log:
  - Total wall time
  - Frames decoded
  - Frames per second throughput
  - Memory high-water mark (via `resource.getrusage`)
- Uses `pytest-benchmark` for statistical rigor
- Output: Markdown table written to `benchmarks/results.md`, committed

**Scope (excluded)**:
- No per-function micro-benchmarks (let Rust phase profile separately)
- No flamegraph generation

**Verification**:
- [ ] Benchmark runs to completion on both test videos
- [ ] Results written to `benchmarks/results.md` with timestamp + git SHA
- [ ] Performance budget: Kino Demo (1m44s) should complete extraction in < 30 seconds on M-series Mac (Python, unoptimized)

**Estimated effort**: 0.5 day

---

### Task 9 — Documentation (`docs/`)

**Goal**: Both user-facing and Rust-translation-facing docs.

**Scope (included)**:
- `docs/algorithm.md` — algorithmic specification with pseudocode, invariants, parameter rationale. Language-agnostic.
- `docs/rust-porting.md` — **the most important doc**:
  - Type map (Python dataclass ↔ Rust struct, field by field)
  - Dependency map (Python lib ↔ Rust crate)
  - Idiom map (Python pattern ↔ Rust pattern) — e.g., `dataclasses.replace` ↔ `Struct { field: new, ..old }`
  - Test fixture map (Python test ↔ Rust test, same JSON fixtures)
  - Known Python-only shortcuts that Rust can tighten (e.g., `np.linspace` vs explicit index loop)
- `README.md` — user-facing: quickstart, example, how to consume scenesdetect output
- `CHANGELOG.md` — standard Keep-a-Changelog format

**Scope (excluded)**:
- No API reference site (docs.rs equivalent) — that's for Rust phase
- No tutorials beyond the quickstart

**Verification**:
- [ ] All Python types have a row in the type map
- [ ] All numpy ops used in the code have a row in the idiom map
- [ ] README quickstart is copy-pasteable and works on a fresh clone
- [ ] Teammate can produce a Rust module skeleton from `rust-porting.md` alone, without reading Python source (sanity-check via code review)

**Estimated effort**: 1 day (continuous, updated as tasks progress)

---

## 4. Implementation Phases

| Phase | Tasks | Deliverable | ETA |
|-------|-------|-------------|-----|
| **P1 — Foundation** | T1 + T2 + T9 (started) | Scaffold + types + type-map docs | 1.5 days |
| **P2 — MVP** | T3 + T4 + T5 (basic) | Working end-to-end on synthetic video | 4.5 days |
| **P3 — Real video** | T7 + T5 (fallback polish) + T8 | CLI + benchmarks on real videos | 2 days |
| **P4 — macOS + polish** | T6 + T9 finalize | Apple Vision saliency + complete docs | 1.5 days |
| **P5 — Handoff** | Teammate review | Comments, Rust translation begins | (parallel to P4) |

**Total Python phase**: ~9–10 working days.
**Rust translation** (by teammate, out of our scope): ~2 weeks separately.

---

## 5. Cross-Cutting Verification

Across all tasks, these must hold:

- [ ] **No OpenCV** anywhere (`grep -r "import cv2"` → 0 hits)
- [ ] **No scipy / sklearn / skimage** in `src/` (test fixtures may use them for generating ground truth, but not production code)
- [ ] **Type hints on every public function** (`mypy --strict` clean)
- [ ] **All public functions have docstrings** with: purpose, args, returns, raises
- [ ] **No global mutable state** (Rust doesn't like it, Python shouldn't need it)
- [ ] **No `from X import *`** (explicit imports only, translation-friendly)
- [ ] **Every test has a fixture file** or is runnable without network

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PyAV API quirks on macOS (FFmpeg 7.x) | Medium | Medium | Pin `av>=13.0,<14.0` — PyAV 13's exception hierarchy (`av.error.FFmpegError` as a common base inheriting from both `OSError` and `ValueError`) is what the CLI's narrowed `except` clause relies on; test on real Kino Demo early |
| Apple Vision pyobjc behavior differs from objc2-vision | Low | Low | Document saliency is "signal only"; Rust may tune independently |
| Numpy precision differs from Rust ndarray (rare but real) | Low | Medium | Golden fixtures at 6-decimal precision with tolerance |
| Shot edge cases: zero-duration, overlapping, negative | Medium | High | Validate inputs strictly; unit tests for pathological shots |
| Decoder returning frames off by ±1 frame from requested PTS | High | Medium | Document ±1 frame tolerance; test at known I-frames |

---

## 7. Out-of-Scope but Noted for Later

Recording decisions that are explicitly **not P1** but likely P2+ (Rust phase or beyond):

- **MMR-based cross-bin deduplication** — if identical shots (news anchors) have visually near-identical keyframes across bins, collapse them. Needs VLM or CLIP embedding, defer.
- **SigLIP medoid as L1 selection** — when selecting 1 frame for indexing (not VLM description), use SigLIP embedding medoid within quality-gated candidates. Needs ONNX runtime integration.
- **Cross-shot coherence** — if consecutive shots form a "scene" in film-theory sense, deduplicate keyframes across shot boundaries. Research needed.
- **Learned quality models** (NIMA, MUSIQ) — if Laplacian variance proves insufficient on real data, revisit.
- **Hardware decode (VideoToolbox)** — Rust phase only.

---

## 8. Glossary

- **Shot**: A contiguous run of video frames from one scene cut to the next, as emitted by scenesdetect.
- **Bucket / Bin**: An equal-duration subdivision of a shot for temporal stratification.
- **Keyframe**: A representative frame extracted from a bucket; may be one of many per shot.
- **Hard gate**: Pass/fail quality threshold that rejects a candidate outright (e.g., black frame).
- **Soft score**: Continuous quality score used to rank candidates that passed the hard gate.
- **Degraded confidence**: Output tag indicating all candidates failed hard gate but one was force-selected.
- **Medoid**: The element of a set with minimum total distance to all others; a "real" (vs interpolated) centroid.

---

## 9. Working Directory Handoff

Current working directory: `/Users/cheongzhiyan/Developer/Findit_app`

**Next steps to switch**:

1. Confirm this task document is complete and accurate (you and I both review).
2. Create the repo: `github.com/Findit-AI/findit-keyframe` (you handle GitHub side; provide local path when ready).
3. Clone locally to e.g. `/Users/cheongzhiyan/Developer/findit-keyframe`.
4. Copy this `TASKS.md` into the new repo root (or `docs/TASKS.md`).
5. Move working directory to the new repo for all subsequent implementation work.

From that point forward, all tasks proceed from the new working directory.
