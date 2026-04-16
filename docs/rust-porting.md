# Rust Porting Guide

> **Audience**: The teammate translating `findit-keyframe` to Rust. **Read this before reading any Python source.** If anything in this doc disagrees with the Python code, the Python code is wrong — file an issue.

This guide intentionally re-derives the public surface so a Rust skeleton can be sketched without consulting the Python implementation. Algorithm details live in [`algorithm.md`](algorithm.md).

## 1. Type Map

| Python (`src/findit_keyframe/`) | Rust target | Notes |
|---------------------------------|-------------|-------|
| `types.Timebase(num: int, den: int)` | `pub struct Timebase { num: i32, den: NonZeroU32 }` | Mirror upstream `scenesdetect::frame::Timebase`. Reduced-form (gcd-based) equality and hash; `den > 0` invariant enforced at construction. |
| `types.Timestamp(pts: int, timebase: Timebase)` | `pub struct Timestamp { pts: i64, timebase: Timebase }` | Match upstream; `seconds() -> f64`. Comparison and hash via integer cross-multiplication so cross-timebase ordering is exact. |
| `types.ShotRange(start: Timestamp, end: Timestamp)` | `pub type ShotRange = scenesdetect::frame::TimeRange;` | Half-open. `end > start` enforced at construction. Reuse the upstream type if the Rust port pulls `scenesdetect` as a dep. |
| `types.SamplingConfig` | `#[derive(Clone, Copy)] pub struct SamplingConfig { ... }` | Plain `Copy` struct; `Default` impl mirrors Python defaults. Mutable in Python; in Rust use struct update syntax. |
| `types.Confidence` (`StrEnum`) | `#[non_exhaustive] pub enum Confidence { High, Low, Degraded }` | `value` is the lowercase string for manifest output; in Rust derive `Display` returning the same. |
| `types.QualityMetrics` | `#[derive(Clone, Copy, PartialEq)] pub struct QualityMetrics { ... }` | All fields `f32` (or `f64` for parity). No `Eq` / `Hash` because of float fields. |
| `types.ExtractedKeyframe` | `pub struct ExtractedKeyframe { ..., rgb: Vec<u8> }` | Mutable in Python; in Rust own the buffer. |
| `decoder.DecodedFrame` | `pub struct DecodedFrame<'a> { pts: Timestamp, width: u32, height: u32, rgb: Cow<'a, [u8]> }` | Borrowed when from decoder buffer; owned after copy. Width/height stored explicitly so `&[u8]` carries no shape info. |
| `decoder.Strategy` (`StrEnum`) | `pub enum Strategy { Sequential, PerShotSeek }` | |
| `quality.QualityGate` | `pub struct QualityGate { min_mean_luma: f32, max_mean_luma: f32, min_luma_variance: f32 }` | `Default` impl matches Python's `15/255`, `240/255`, `5.0`. `passes(&QualityMetrics) -> bool`. |
| `saliency.SaliencyProvider` (`Protocol`) | `pub trait SaliencyProvider { fn compute(&self, rgb: &RgbFrame) -> f32; }` | Object-safe trait. |
| `saliency.NoopSaliencyProvider` | `pub struct NoopSaliencyProvider;` | Returns `0.0`. |
| `saliency.AppleVisionSaliencyProvider` | `pub struct AppleVisionSaliencyProvider { ... }` (cfg `target_os = "macos"`) | Wraps `objc2-vision`'s `VNGenerateAttentionBasedSaliencyImageRequest`. |

## 2. Dependency Map

| Python | Rust |
|--------|------|
| `numpy` | Plain `Vec<f32>` / `&[u8]`. **Do not** use `ndarray`; upstream `scenesdetect` rejected it. SIMD via `std::simd` or hand-tuned `aarch64`/`x86`/`wasm32` modules behind `cfg`. |
| `av` (PyAV) | `ffmpeg-next` (or upstream `scenesdetect`'s decoder helpers if exposed). VideoToolbox hwaccel via `ffmpeg-sys` + `AVHWDeviceType::VideoToolbox`. |
| `pyobjc-framework-Vision` + `pyobjc-framework-Quartz` | `objc2-vision`, `objc2-quartz-core`, `objc2-core-foundation`. |
| `argparse` | `clap` (derive macros). |
| `pytest` | Cargo built-in `#[test]` + `proptest` for property tests. |
| `pytest-benchmark` | `criterion` (already used by `scenesdetect`). |
| `json` (stdlib) | `serde` + `serde_json`. |
| `dataclasses.replace` | Struct update syntax: `SamplingConfig { foo: new_foo, ..old }`. |
| PyAV `av.open(..., format="image2")` + `mjpeg` codec for JPEG output | `image::codecs::jpeg::JpegEncoder` with quality 92. |

## 3. Idiom Map

### Operator-level mappings

| Python pattern | Rust pattern |
|----------------|--------------|
| `@dataclass(frozen=True)` | `#[derive(Clone, Copy, PartialEq, Eq, Hash)] pub struct ...` (drop `Eq`/`Hash` for floats). |
| `Optional[T]` / `T \| None` | `Option<T>`. |
| `list[T]` | `Vec<T>` for owned, `&[T]` for borrowed. |
| `Iterator[(int, X)]` | `impl Iterator<Item = (usize, X)>`. |
| `Protocol` | `trait`. |
| `enum.StrEnum` | `enum` with `#[non_exhaustive]` for public; impl `Display` for the lowercase string. |
| `dataclasses.replace(c, x=v)` | `Config { x: v, ..c }`. |

### Numpy ops we actually use

| Python | Rust |
|--------|------|
| `rgb[..., 0].astype(np.uint32)` etc. (per-channel split for BT.601) | Manual `for i in (0..len).step_by(3)`; promote to `u32` via `as u32`. |
| `((66*r + 129*g + 25*b + 128) >> 8) + 16` | Same expression, `u32` arithmetic, `>> 8` then `+ 16`. |
| 3×3 Laplacian via slicing: `f[:-2,1:-1] + f[2:,1:-1] + f[1:-1,:-2] + f[1:-1,2:] - 4*f[1:-1,1:-1]` | Index loop over `(y, x)` for `y in 1..h-1`, `x in 1..w-1`; reads centre + 4 neighbours, accumulates `i32`. |
| `arr.var()` (population) | Two-pass: mean = `sum / n`, var = `sum((x-mean)^2) / n` (or Welford). |
| `arr.var(ddof=1)` (sample) | Same but divide by `n - 1`. |
| `np.histogram(luma, bins=256, range=(0, 256))` | `[u32; 256]` accumulator; one pass over pixels. (Upstream `scenesdetect::histogram` uses 4-wide parallel accumulators — borrow that pattern.) |
| Shannon entropy: `-sum(p * log2(p)) for p > 0` | Manual loop; `f64::log2`. |
| `np.argsort(kind="stable")` for ordinal rank | `Vec<(idx, value)>::sort_by(...)` (Rust's `sort` is stable). |
| `np.argmax(scores)` | `scores.iter().enumerate().max_by(...).map(|(i, _)| i)`. |
| `np.frombuffer(bytes, dtype=np.uint8).reshape(...)` | Just a slice view — no copy. |

### Apple Vision idiom

| Python | Rust (`objc2-vision`) |
|--------|----------------------|
| RGB → padded RGBX (32-bit aligned) → `CGImageCreate` | Same: `CGImageCreate` wants 32 bpp; pad to RGBA with `kCGImageAlphaNoneSkipLast`. |
| `VNImageRequestHandler.alloc().initWithCGImage_options_(cg, {})` | `VNImageRequestHandler::initWithCGImage_options:` |
| `VNGenerateAttentionBasedSaliencyImageRequest.alloc().init()` | `VNGenerateAttentionBasedSaliencyImageRequest::new()`. |
| `handler.performRequests_error_([req], None)` | `handler.performRequests(&[req])?` (returns `Result`). |
| `obs.salientObjects()` returns `[VNRectangleObservation]`; sum `area * confidence` | Same iteration, sum `f32`. |

## 4. Test Fixture Map

JSON / image fixtures under `tests/fixtures/` (where present) are the **shared ground truth** between implementations. Tests that build inputs procedurally (most of T2-T5) should be replayed in Rust with the same numeric expectations.

| Python test | Rust test (target) | Notes |
|-------------|--------------------|-------|
| `tests/test_types.py` | `tests/types_*.rs` | Hand-derived numeric expectations; replay verbatim. |
| `tests/test_quality.py` | `tests/quality_*.rs` | All numeric assertions are derived in test docstrings; replay verbatim. |
| `tests/test_decoder.py` | `tests/decoder_*.rs` | Tiny test video encoded inside `conftest.py`; reproduce with the same encoder settings. |
| `tests/test_sampler.py` | `tests/sampler_*.rs` | Pure-function tests + integration on `varied_video` (per-frame mid-tone noise; reproduce). |
| `tests/test_cli.py` | `tests/cli_*.rs` | Manifest schema and exit codes are the contract. |
| `tests/test_saliency.py` | `tests/saliency_*.rs` | Apple Vision tests are macOS-gated by `cfg(target_os = "macos")`. |

Tolerance: 6 decimal places for float comparisons unless documented otherwise.

## 5. Python-Only Shortcuts (Tighten in Rust)

- `np.var(ddof=1)`: numpy uses an O(n) two-pass internally; Rust should consider Welford for numerical stability on large `n`.
- `_ordinal_rank` does an `argsort` to get ranks. Rust can use `sort_by_cached_key` or compute ranks in-place during merge-sort for slightly less allocation.
- The Apple Vision provider holds Python-side strong refs to factory classes for hot-path lookup avoidance. Rust gets this for free via static linkage.
- `_select_with_fallback` re-decodes candidates in the expanded window without checking for overlap with the native probe set. Rust can dedup by PTS to skip a few decodes per fallback bin.

## 6. Known Divergence Risk

| Risk | Mitigation |
|------|------------|
| Numpy float reduction order vs Rust scalar order | Cross-validate at 6 decimals; document any operator that uses tree-reduction in numpy. |
| PyAV resize (swscale) vs Rust resize | Pin both to bilinear; document. |
| Apple Vision saliency mass: pyobjc returns confidence-weighted bounding boxes; `objc2-vision` returns the same observations but the iteration API differs slightly | Test on identical input; tolerate ±1 % saliency mass. |
| `>> 8` truncation in `rgb_to_luma` differs from float division | Both implementations must use integer fixed-point exactly per spec. Test `pure_green` round-trip (33023 >> 8 = 128 → Y = 144) catches this. |

## 7. Translation Workflow

1. Read this doc.
2. For each Python module, generate a Rust module skeleton with matching public types and stub function signatures.
3. Port one module at a time, in this order: `types` → `quality` → `decoder` → `sampler` → `saliency` → `cli`.
4. After each module, run the equivalent Rust tests; expectations come from the Python tests verbatim.
5. When a numerical assertion diverges, **first** check this doc's "Known Divergence Risk" table; **then** open an issue.
