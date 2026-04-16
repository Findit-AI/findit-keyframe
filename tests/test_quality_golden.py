"""Bit-precision regression contract for :func:`findit_keyframe.quality.compute_quality`.

The JSON fixture at ``tests/fixtures/quality/canonical.json`` encodes both
the input frame specifications (purely deterministic, integer-arithmetic
generators — no PRNG, so a Rust port can reproduce them byte-for-byte) and
the expected :class:`QualityMetrics` rounded to 6 decimal places.

The Rust translation must replay this fixture: load the JSON, materialise
each frame using the same generator semantics (see ``_build_frame``), run
its own ``compute_quality`` and assert each field matches expected to
``1e-6``.

To regenerate the fixture after an intentional algorithm change, run::

    python tests/test_quality_golden.py

Review the diff in ``canonical.json`` carefully before committing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from findit_keyframe.quality import compute_quality

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "quality" / "canonical.json"
TOLERANCE = 1e-6


# --------------------------------------------------------------------------- #
# Frame generators (pure integer arithmetic, no PRNG)                         #
# --------------------------------------------------------------------------- #


def _build_frame(spec: dict) -> np.ndarray:
    """Materialise a frame from its declarative ``spec`` dict.

    Generator catalogue (exhaustive — the Rust port re-implements these
    same six kinds against the same JSON inputs):

    ``solid`` — ``{rgb: [R, G, B], size: [H, W]}``
        Every pixel equals ``rgb``.

    ``channel`` — ``{channel: "r" | "g" | "b", value: V, size: [H, W]}``
        Selected channel = ``V``; the other two channels = 0.

    ``h_gradient`` — ``{low: L, high: H, size: [H, W]}``
        ``pixel[y, x, c] = L + (H - L) * x // (W - 1)``. Integer division;
        identical R, G, B per pixel.

    ``v_gradient`` — same as ``h_gradient`` but along the ``y`` axis.

    ``checker`` — ``{cell: K, low: L, high: H, size: [H, W]}``
        ``pixel[y, x, c] = H if ((y // K) + (x // K)) % 2 == 0 else L``;
        identical R, G, B per pixel.

    ``single_pixel`` —
    ``{bg: [B, B, B], fg: [F, F, F], position: [py, px], size: [H, W]}``
        Background ``bg`` everywhere, foreground ``fg`` at ``(py, px)``.
    """
    height, width = spec["size"]
    kind = spec["kind"]

    if kind == "solid":
        r, g, b = spec["rgb"]
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[..., 0] = r
        frame[..., 1] = g
        frame[..., 2] = b
        return frame

    if kind == "channel":
        channel_idx = {"r": 0, "g": 1, "b": 2}[spec["channel"]]
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., channel_idx] = spec["value"]
        return frame

    if kind == "h_gradient":
        low, high = spec["low"], spec["high"]
        ramp = np.array(
            [low + (high - low) * x // (width - 1) for x in range(width)],
            dtype=np.uint8,
        )
        frame = np.empty((height, width, 3), dtype=np.uint8)
        for c in range(3):
            frame[..., c] = ramp[None, :]
        return frame

    if kind == "v_gradient":
        low, high = spec["low"], spec["high"]
        ramp = np.array(
            [low + (high - low) * y // (height - 1) for y in range(height)],
            dtype=np.uint8,
        )
        frame = np.empty((height, width, 3), dtype=np.uint8)
        for c in range(3):
            frame[..., c] = ramp[:, None]
        return frame

    if kind == "checker":
        cell = spec["cell"]
        low, high = spec["low"], spec["high"]
        frame = np.full((height, width, 3), low, dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if ((y // cell) + (x // cell)) % 2 == 0:
                    frame[y, x] = high
        return frame

    if kind == "single_pixel":
        bg = spec["bg"]
        fg = spec["fg"]
        py, px = spec["position"]
        frame = np.empty((height, width, 3), dtype=np.uint8)
        for c in range(3):
            frame[..., c] = bg[c]
        for c in range(3):
            frame[py, px, c] = fg[c]
        return frame

    raise ValueError(f"unknown spec kind: {kind!r}")


# --------------------------------------------------------------------------- #
# Test                                                                        #
# --------------------------------------------------------------------------- #


def test_canonical_quality_metrics_match_golden_fixture():
    """Every field of every canonical frame must match expected to 6 decimals.

    A failure means ``compute_quality``'s output has shifted at the bit
    level. If the shift is intentional, regenerate the fixture
    (``python tests/test_quality_golden.py``) and audit the JSON diff
    before committing.
    """
    data = json.loads(FIXTURE_PATH.read_text())
    assert data["version"] == 1, f"unsupported fixture version: {data['version']}"
    assert len(data["frames"]) >= 10, "fixture must contain at least 10 canonical frames"

    failures: list[str] = []
    for entry in data["frames"]:
        frame = _build_frame(entry["spec"])
        actual = compute_quality(frame)
        actual_dict = asdict(actual)
        for field, expected in entry["expected"].items():
            got = actual_dict[field]
            if abs(got - expected) > TOLERANCE:
                failures.append(f"  {entry['id']}.{field}: expected {expected:.6f}, got {got:.10f}")
    if failures:
        raise AssertionError("Golden fixture mismatch:\n" + "\n".join(failures))


# --------------------------------------------------------------------------- #
# Regenerator (run directly: ``python tests/test_quality_golden.py``)         #
# --------------------------------------------------------------------------- #


_CANONICAL_SPECS: list[tuple[str, dict]] = [
    ("solid_black", {"kind": "solid", "rgb": [0, 0, 0], "size": [32, 32]}),
    ("solid_white", {"kind": "solid", "rgb": [255, 255, 255], "size": [32, 32]}),
    ("solid_gray_128", {"kind": "solid", "rgb": [128, 128, 128], "size": [32, 32]}),
    ("channel_red", {"kind": "channel", "channel": "r", "value": 255, "size": [32, 32]}),
    ("channel_green", {"kind": "channel", "channel": "g", "value": 255, "size": [32, 32]}),
    ("channel_blue", {"kind": "channel", "channel": "b", "value": 255, "size": [32, 32]}),
    ("h_gradient_0_255_64w", {"kind": "h_gradient", "low": 0, "high": 255, "size": [32, 64]}),
    ("v_gradient_0_255_64h", {"kind": "v_gradient", "low": 0, "high": 255, "size": [64, 32]}),
    ("checker_8px_0_255", {"kind": "checker", "cell": 8, "low": 0, "high": 255, "size": [32, 32]}),
    (
        "single_pixel_100_in_5x5",
        {
            "kind": "single_pixel",
            "bg": [0, 0, 0],
            "fg": [100, 100, 100],
            "position": [2, 2],
            "size": [5, 5],
        },
    ),
]


def _regenerate_fixture() -> None:
    """Recompute every canonical frame's metrics and rewrite the JSON file.

    Intended for human-driven runs after an intentional algorithm change.
    """
    frames = []
    for frame_id, spec in _CANONICAL_SPECS:
        frame = _build_frame(spec)
        metrics = compute_quality(frame)
        frames.append(
            {
                "id": frame_id,
                "spec": spec,
                "expected": {k: round(float(v), 6) for k, v in asdict(metrics).items()},
            }
        )
    data = {
        "version": 1,
        "purpose": (
            "Bit-precision regression contract for compute_quality. "
            "Generator semantics live in tests/test_quality_golden.py::_build_frame "
            "and the Rust port replays this fixture against its own implementation."
        ),
        "tolerance": TOLERANCE,
        "frames": frames,
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {len(frames)} frames to {FIXTURE_PATH}")


if __name__ == "__main__":
    _regenerate_fixture()
