"""Tests for ``findit_keyframe.decoder``.

The pure-logic pieces (``Strategy`` / ``pick_strategy``) get bare numeric
tests. The PyAV-backed pieces use a session-scoped fixture (``tiny_video``)
encoding a 1-second 30-fps ramp so we exercise real seek/decode paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from findit_keyframe.decoder import (
    DecodedFrame,
    Strategy,
    VideoDecoder,
    pick_strategy,
)
from findit_keyframe.types import ShotRange, Timebase, Timestamp

if TYPE_CHECKING:
    from pathlib import Path


def _ts(seconds: float, tb: Timebase) -> Timestamp:
    """Build a Timestamp for ``seconds`` in ``tb``."""
    return Timestamp(round(seconds * tb.den / tb.num), tb)


def _make_back_to_back_shots(n: int, duration: float) -> list[ShotRange]:
    tb = Timebase(1, 1000)
    span = duration / n
    return [
        ShotRange(
            start=Timestamp(round(i * span * 1000), tb),
            end=Timestamp(round((i + 1) * span * 1000), tb),
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# pick_strategy                                                               #
# --------------------------------------------------------------------------- #


class TestPickStrategy:
    def test_empty_shots_returns_per_shot_seek(self):
        assert pick_strategy([], 60.0) is Strategy.PerShotSeek

    def test_zero_duration_returns_per_shot_seek(self):
        assert pick_strategy(_make_back_to_back_shots(10, 60.0), 0.0) is Strategy.PerShotSeek

    def test_low_density_returns_per_shot_seek(self):
        # 10 shots / 60 s = 0.166 shots/s, well below 0.3 threshold.
        assert pick_strategy(_make_back_to_back_shots(10, 60.0), 60.0) is Strategy.PerShotSeek

    def test_high_density_returns_sequential(self):
        # 30 shots / 60 s = 0.5 shots/s, above threshold.
        assert pick_strategy(_make_back_to_back_shots(30, 60.0), 60.0) is Strategy.Sequential

    def test_huge_count_returns_sequential(self):
        # 250 shots / 1000 s = 0.25 shots/s (below density), but count > 200.
        assert pick_strategy(_make_back_to_back_shots(250, 1000.0), 1000.0) is Strategy.Sequential

    def test_density_threshold_is_strict(self):
        # Threshold is `> 0.3`, exclusive. Exactly 0.3 stays at PerShotSeek.
        assert pick_strategy(_make_back_to_back_shots(30, 100.0), 100.0) is Strategy.PerShotSeek


# --------------------------------------------------------------------------- #
# VideoDecoder — open & metadata                                              #
# --------------------------------------------------------------------------- #


class TestVideoDecoderOpen:
    def test_metadata(self, tiny_video: Path):
        with VideoDecoder.open(tiny_video) as dec:
            assert dec.fps == pytest.approx(30.0, abs=0.5)
            assert dec.duration_sec == pytest.approx(1.0, abs=0.1)
            assert dec.width == 64
            assert dec.height == 64

    def test_target_size_resizes(self, tiny_video: Path):
        with VideoDecoder.open(tiny_video, target_size=32) as dec:
            assert dec.width == 32
            assert dec.height == 32
            f = dec.decode_at(0.0)
            assert f.rgb.shape == (32, 32, 3)
            assert f.rgb.dtype == np.uint8

    def test_close_releases_container(self, tiny_video: Path):
        dec = VideoDecoder.open(tiny_video)
        dec.close()
        # Subsequent decode must fail because the container is closed.
        with pytest.raises(Exception):  # noqa: B017, PT011 — PyAV raises various
            dec.decode_at(0.0)


# --------------------------------------------------------------------------- #
# VideoDecoder — decode_at                                                    #
# --------------------------------------------------------------------------- #


class TestDecodeAt:
    def test_first_frame_returns_decoded_frame(self, tiny_video: Path):
        with VideoDecoder.open(tiny_video) as dec:
            f = dec.decode_at(0.0)
            assert isinstance(f, DecodedFrame)
            assert f.rgb.shape == (64, 64, 3)
            assert f.rgb.dtype == np.uint8

    def test_first_frame_pts_near_zero(self, tiny_video: Path):
        with VideoDecoder.open(tiny_video) as dec:
            f = dec.decode_at(0.0)
            assert f.pts.seconds < 1.0 / 30.0

    def test_mid_frame_pts_within_one_frame(self, tiny_video: Path):
        with VideoDecoder.open(tiny_video) as dec:
            target = 0.5
            f = dec.decode_at(target)
            assert abs(f.pts.seconds - target) < 1.0 / 30.0 + 1e-3

    def test_decoded_frame_dimensions_match_target_size(self, tiny_video: Path):
        with VideoDecoder.open(tiny_video, target_size=32) as dec:
            f = dec.decode_at(0.5)
            assert f.width == 32
            assert f.height == 32
            assert f.rgb.shape == (32, 32, 3)

    def test_seek_past_end_raises(self, tiny_video: Path):
        with VideoDecoder.open(tiny_video) as dec, pytest.raises(ValueError, match="Could not"):
            dec.decode_at(10.0)


# --------------------------------------------------------------------------- #
# VideoDecoder — decode_sequential                                            #
# --------------------------------------------------------------------------- #


class TestDecodeSequential:
    def test_empty_shots_yields_nothing(self, tiny_video: Path):
        with VideoDecoder.open(tiny_video) as dec:
            assert list(dec.decode_sequential([])) == []

    def test_full_coverage_yields_every_frame(self, tiny_video: Path):
        with VideoDecoder.open(tiny_video) as dec:
            shots = [ShotRange(start=_ts(0.0, dec.timebase), end=_ts(2.0, dec.timebase))]
            frames = list(dec.decode_sequential(shots))
        # Allow ±2 frames slack for VFR-ish PTS quantisation at the boundary.
        assert 28 <= len(frames) <= 32
        assert all(shot_id == 0 for shot_id, _ in frames)
        # Frames are emitted in PTS order.
        ptses = [f.pts.seconds for _, f in frames]
        assert ptses == sorted(ptses)

    def test_disjoint_shots_yield_subsets_with_correct_ids(self, tiny_video: Path):
        with VideoDecoder.open(tiny_video) as dec:
            tb = dec.timebase
            shots = [
                ShotRange(start=_ts(0.0, tb), end=_ts(0.3, tb)),  # ~9 frames
                ShotRange(start=_ts(0.6, tb), end=_ts(0.8, tb)),  # ~6 frames
            ]
            frames = list(dec.decode_sequential(shots))
        ids = {shot_id for shot_id, _ in frames}
        assert ids == {0, 1}
        # No frame should fall in the [0.3, 0.6) gap.
        for shot_id, f in frames:
            t = f.pts.seconds
            if shot_id == 0:
                assert 0.0 <= t < 0.3 + 1e-3
            else:
                assert 0.6 - 1e-3 <= t < 0.8 + 1e-3

    def test_unsorted_shot_input_handled(self, tiny_video: Path):
        # Original shot indices must be preserved when shots are passed out of order.
        with VideoDecoder.open(tiny_video) as dec:
            tb = dec.timebase
            shots = [
                ShotRange(start=_ts(0.6, tb), end=_ts(0.8, tb)),  # id 0, later in time
                ShotRange(start=_ts(0.0, tb), end=_ts(0.3, tb)),  # id 1, earlier
            ]
            frames = list(dec.decode_sequential(shots))
        # Frames at t < 0.3 must be tagged with shot id 1 (the earlier one).
        for shot_id, f in frames:
            if f.pts.seconds < 0.3:
                assert shot_id == 1
            elif 0.6 <= f.pts.seconds < 0.8:
                assert shot_id == 0
