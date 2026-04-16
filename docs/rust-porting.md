# Rust Porting Guide

> **Audience**: The teammate translating `findit-keyframe` to Rust. **Read this before reading any Python source.** If anything in this doc disagrees with the Python code, the Python code is wrong — file an issue.

This guide intentionally re-derives the public surface so a Rust skeleton can be sketched without consulting the Python implementation. Algorithm details live in [`algorithm.md`](algorithm.md).

## 1. Type Map (filled during T2)

| Python (`src/findit_keyframe/types.py`) | Rust target | Notes |
|-----------------------------------------|-------------|-------|
| `Timebase(num: int, den: int)` | `pub struct Timebase { num: i32, den: NonZeroU32 }` | Mirror upstream `scenesdetect::frame::Timebase`. Reduced-form equality, 128-bit cross-multiply for `Ord`. |
| `Timestamp(pts: int, timebase: Timebase)` | `pub struct Timestamp { pts: i64, timebase: Timebase }` | Match upstream; `seconds()` returns `f64`. |
| `ShotRange(start: Timestamp, end: Timestamp)` | `pub type ShotRange = scenesdetect::frame::TimeRange;` | Half-open. Reuse the upstream type if Rust port pulls `scenesdetect` as a dep. |
| `SamplingConfig` | `pub struct SamplingConfig { ... }` | Plain `Copy` struct of `f32`/`u32`; `Default` impl mirrors Python defaults. |
| `Confidence` enum | `#[non_exhaustive] pub enum Confidence { High, Low, Degraded }` | |
| `QualityMetrics` | `#[derive(Clone, Copy, PartialEq)] pub struct QualityMetrics { ... }` | All fields `f32`. |
| `ExtractedKeyframe` | `pub struct ExtractedKeyframe<'a> { ..., rgb: Cow<'a, [u8]> }` | Borrowed when from decoder buffer; owned after copy. |

(Rows added as types land in `types.py`.)

## 2. Dependency Map

| Python | Rust |
|--------|------|
| `numpy` | Plain `Vec<f32>` / `&[u8]`. **Do not** use `ndarray`; upstream `scenesdetect` rejected it. SIMD via `std::simd` or hand-tuned `aarch64`/`x86`/`wasm32` modules behind `cfg`. |
| `av` (PyAV) | `ffmpeg-next` (or upstream `scenesdetect`'s decoder helpers if exposed). VideoToolbox hwaccel via `ffmpeg-sys` + `AVHWDeviceType::VideoToolbox`. |
| `pyobjc-framework-Vision` | `objc2-vision` + `objc2-core-foundation`. `VNGenerateAttentionBasedSaliencyImageRequest` is the same call. |
| `argparse` | `clap` (derive macros). |
| `pytest` | Cargo built-in `#[test]` + `proptest` for property tests. |
| `pytest-benchmark` | `criterion` (already used by `scenesdetect`). |
| `json` | `serde` + `serde_json`. |
| `dataclasses.replace` | Struct update syntax: `SamplingConfig { foo: new_foo, ..old }`. |

## 3. Idiom Map

| Python pattern | Rust pattern |
|----------------|--------------|
| `@dataclass(frozen=True)` | `#[derive(Clone, Copy, PartialEq, Eq, Hash)] pub struct ...` (drop `Eq`/`Hash` for floats). |
| `Optional[T]` | `Option<T>`. |
| `list[T]` | `Vec<T>` for owned, `&[T]` for borrowed. |
| `Iterator[(int, X)]` | `impl Iterator<Item = (usize, X)>`. |
| `Protocol` | `trait`. Default impls go on the trait. |
| `np.linspace(a, b, n)` | Explicit index loop: `(0..n).map(|i| a + (b - a) * i as f32 / (n - 1) as f32)`. |
| `np.histogram(luma, bins=256)` | `[u32; 256]` accumulator, single pass. (Upstream `scenesdetect::histogram` uses 4-wide parallel accumulators — borrow that.) |
| `np.mean(arr)` / `np.var(arr)` | Manual `iter().sum()` + `len()`; Welford for variance if numerical issues appear. |
| `enum.Enum` | `enum` with `#[non_exhaustive]` for public ones. |

## 4. Test Fixture Map

JSON fixtures under `tests/fixtures/` are the **shared ground truth** between implementations.

| Python test | Rust test (target) | Fixture |
|-------------|--------------------|---------|
| `tests/test_quality_golden.py` | `tests/quality_golden.rs` | `tests/fixtures/quality/*.json` |
| `tests/test_sampler_kino_demo.py` | `tests/sampler_kino_demo.rs` | `tests/fixtures/sampler/kino_demo.json` |
| `tests/test_types_*.py` | `tests/types_*.rs` | None (algebraic) |

Tolerance: 6 decimal places for floats unless documented otherwise.

## 5. Python-Only Shortcuts (Tighten in Rust)

Track here as they emerge during implementation:

- _(none yet)_

## 6. Known Divergence Risk

| Risk | Mitigation |
|------|------------|
| Numpy float reduction order vs Rust scalar order | Cross-validate at 6 decimals; document any operator that uses tree-reduction in numpy. |
| PyAV resize (swscale) vs Rust resize | Pin both to bilinear; document. |
| Apple Vision saliency mass: pyobjc returns `Quartz.CIImage` — pixel readback must match `objc2-vision` `CGImage` path. | Test on identical input; tolerate ±1 % saliency mass. |

## 7. Translation Workflow

1. Read this doc.
2. For each Python module, generate a Rust module skeleton with matching public types and stub function signatures.
3. Port one module at a time, in this order: `types` → `quality` → `decoder` → `sampler` → `saliency` → `cli`.
4. After each module, run the shared JSON fixtures from `tests/fixtures/`.
5. When a fixture diverges, **first** check this doc's "Known Divergence Risk" table; **then** open an issue.
