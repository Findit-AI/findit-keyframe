"""Deterministic regression snapshot for :func:`extract_all`.

This file stands in for the *Kino Demo regression fixture* called out in
``TASKS.md`` T5: a JSON snapshot of ``(shot_id, bin_index, timestamp,
quality)`` tuples that fails the moment the algorithm's selection or
scoring drifts. Until the real asset is available, we use an **in-memory
fake decoder** that returns deterministic noise frames so the snapshot is
bit-exact across machines and PyAV / FFmpeg versions.

To regenerate after an intentional algorithm change, run::

    python tests/test_sampler_regression.py

Review the diff in the JSON file before committing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from findit_keyframe.decoder import DecodedFrame
from findit_keyframe.sampler import extract_all
from findit_keyframe.types import SamplingConfig, ShotRange, Timebase, Timestamp

if TYPE_CHECKING:
    from collections.abc import Callable

SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "regression" / "extract_all_synthetic.json"


# --------------------------------------------------------------------------- #
# In-memory deterministic decoder                                             #
# --------------------------------------------------------------------------- #


@dataclass
class _FakeDecoder:
    """Duck-typed VideoDecoder for bit-exact regression testing.

    The sampler's only contact with the decoder is :attr:`duration_sec`,
    :attr:`timebase`, and :meth:`decode_at`. We satisfy that surface
    without touching PyAV so the snapshot is unaffected by codec build
    differences across machines.
    """

    duration_sec: float
    fps: int
    frame_fn: Callable[[int], np.ndarray]

    @property
    def timebase(self) -> Timebase:
        return Timebase(1, self.fps)

    def decode_at(self, time_sec: float) -> DecodedFrame:
        max_idx = int(self.duration_sec * self.fps) - 1
        idx = max(0, min(max_idx, round(time_sec * self.fps)))
        rgb = self.frame_fn(idx)
        height, width = rgb.shape[:2]
        return DecodedFrame(
            pts=Timestamp(idx, self.timebase),
            width=width,
            height=height,
            rgb=rgb,
        )


def _synthetic_noise_frame(idx: int) -> np.ndarray:
    """Deterministic mid-tone noise frame keyed by frame index."""
    rng = np.random.default_rng(seed=idx)
    return rng.integers(50, 201, size=(64, 64, 3), dtype=np.uint8)


def _build_synthetic_decoder() -> _FakeDecoder:
    """A 5-second 30 fps fake decoder yielding deterministic noise."""
    return _FakeDecoder(duration_sec=5.0, fps=30, frame_fn=_synthetic_noise_frame)


def _shot(start_sec: float, end_sec: float, tb: Timebase) -> ShotRange:
    return ShotRange(
        start=Timestamp(round(start_sec * tb.den / tb.num), tb),
        end=Timestamp(round(end_sec * tb.den / tb.num), tb),
    )


def _run_extract_all() -> list[dict]:
    """Run the algorithm on a fixed input and flatten to a JSON-serialisable list.

    Inputs are chosen to exercise both code paths simultaneously:

    * shot 0 — short, single-bin (no within-bin contention).
    * shot 1 — longer, multi-bin (exercises scoring + cross-bin ordering).
    """
    decoder = _build_synthetic_decoder()
    tb = decoder.timebase
    shots = [
        _shot(0.0, 0.4, tb),
        _shot(0.5, 4.5, tb),
    ]
    config = SamplingConfig(target_interval_sec=1.0, target_size=64)
    results = extract_all(shots, decoder, config)
    return [
        {
            "shot_id": kf.shot_id,
            "bucket": kf.bucket_index,
            "timestamp_sec": round(kf.timestamp.seconds, 6),
            "confidence": kf.confidence.value,
            "quality": {k: round(float(v), 6) for k, v in asdict(kf.quality).items()},
        }
        for shot_keyframes in results
        for kf in shot_keyframes
    ]


# --------------------------------------------------------------------------- #
# Test                                                                        #
# --------------------------------------------------------------------------- #


def test_extract_all_matches_regression_snapshot():
    """Every field of every emitted keyframe must match the JSON snapshot.

    This catches: bin partitioning changes, candidate timestamp shifts,
    scoring weight or normalisation changes, fallback ordering changes.
    PyAV / FFmpeg cannot be blamed because the input pipeline is in-memory.
    """
    actual = _run_extract_all()
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    assert snapshot["version"] == 1, f"unsupported snapshot version: {snapshot['version']}"
    assert actual == snapshot["entries"], (
        "extract_all output diverged from snapshot. If intentional, "
        "regenerate via `python tests/test_sampler_regression.py` and "
        "audit the diff before committing."
    )


# --------------------------------------------------------------------------- #
# Regenerator (run directly: `python tests/test_sampler_regression.py`)       #
# --------------------------------------------------------------------------- #


def _regenerate_snapshot() -> None:
    entries = _run_extract_all()
    data = {
        "version": 1,
        "purpose": (
            "Deterministic regression snapshot for extract_all over a "
            "synthetic in-memory decoder. Stand-in for the Kino Demo "
            "regression fixture in TASKS.md T5; replace when the real "
            "asset becomes available."
        ),
        "input": {
            "decoder": "in-memory _FakeDecoder, 5.0 s @ 30 fps, 64x64 noise frames",
            "shots": ["[0.0, 0.4) s", "[0.5, 4.5) s"],
            "config": "SamplingConfig(target_interval_sec=1.0, target_size=64)",
        },
        "entries": entries,
    }
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {len(entries)} entries to {SNAPSHOT_PATH}")


if __name__ == "__main__":
    _regenerate_snapshot()
