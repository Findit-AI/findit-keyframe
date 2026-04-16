"""Tests for ``findit_keyframe.sampler``.

Pure-function tests (binning, scoring, selection) get bare numeric checks.
Integration tests use the ``varied_video`` and ``tiny_video`` fixtures so
the fallback path (low/degraded confidence) is exercised on a real decoder.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pytest

from findit_keyframe.decoder import DecodedFrame, VideoDecoder
from findit_keyframe.quality import QualityGate
from findit_keyframe.sampler import (
    _candidate_times,
    _ordinal_rank,
    compute_bins,
    extract_all,
    extract_for_shot,
    score_bin_candidates,
    select_from_bin,
)
from findit_keyframe.types import (
    Confidence,
    ExtractedKeyframe,
    QualityMetrics,
    SamplingConfig,
    ShotRange,
    Timebase,
    Timestamp,
)

if TYPE_CHECKING:
    from pathlib import Path


def _shot(start_sec: float, end_sec: float) -> ShotRange:
    tb = Timebase(1, 1000)
    return ShotRange(
        start=Timestamp(round(start_sec * 1000), tb),
        end=Timestamp(round(end_sec * 1000), tb),
    )


def _qm(**overrides: float) -> QualityMetrics:
    defaults = {
        "laplacian_var": 100.0,
        "mean_luma": 0.5,
        "luma_variance": 1000.0,
        "entropy": 7.0,
        "saliency_mass": 0.0,
    }
    defaults.update(overrides)
    return QualityMetrics(**defaults)


def _solid_frame(pts_sec: float, gray: int = 128, size: int = 32) -> DecodedFrame:
    rgb = np.full((size, size, 3), gray, dtype=np.uint8)
    return DecodedFrame(
        pts=Timestamp(round(pts_sec * 1000), Timebase(1, 1000)),
        width=size,
        height=size,
        rgb=rgb,
    )


def _noise_frame(pts_sec: float, seed: int, size: int = 32) -> DecodedFrame:
    rng = np.random.default_rng(seed=seed)
    rgb = rng.integers(50, 201, size=(size, size, 3), dtype=np.uint8)
    return DecodedFrame(
        pts=Timestamp(round(pts_sec * 1000), Timebase(1, 1000)),
        width=size,
        height=size,
        rgb=rgb,
    )


# --------------------------------------------------------------------------- #
# compute_bins                                                                #
# --------------------------------------------------------------------------- #


class TestComputeBins:
    def test_short_shot_yields_one_bin(self):
        # D = 1s, I = 4s -> N = ceil(1/4) = 1.
        bins = compute_bins(_shot(0.0, 1.0), SamplingConfig())
        assert len(bins) == 1

    def test_typical_shot_yields_n_bins(self):
        # D = 60s, I = 4s -> N = 15.
        bins = compute_bins(_shot(0.0, 60.0), SamplingConfig())
        assert len(bins) == 15

    def test_long_shot_capped_at_max(self):
        # D = 120s, I = 4s -> N = 30, capped at max_frames_per_shot = 16.
        bins = compute_bins(_shot(0.0, 120.0), SamplingConfig())
        assert len(bins) == 16

    def test_two_bin_case_documented(self):
        # D = 5s, I = 4s -> N = ceil(5/4) = 2 (per TASKS.md verification).
        bins = compute_bins(_shot(0.0, 5.0), SamplingConfig())
        assert len(bins) == 2

    def test_bins_are_contiguous_after_shrink(self):
        bins = compute_bins(_shot(10.0, 70.0), SamplingConfig())
        for (_, end), (start, _) in pairwise(bins):
            assert end == pytest.approx(start)

    def test_bins_are_equal_width(self):
        bins = compute_bins(_shot(0.0, 60.0), SamplingConfig())
        widths = [b - a for a, b in bins]
        assert all(w == pytest.approx(widths[0]) for w in widths)

    def test_first_bin_starts_after_shrink(self):
        cfg = SamplingConfig()
        bins = compute_bins(_shot(0.0, 60.0), cfg)
        expected_first_start = 0.0 + cfg.boundary_shrink_pct * 60.0
        assert bins[0][0] == pytest.approx(expected_first_start)

    def test_last_bin_ends_before_shrink(self):
        cfg = SamplingConfig()
        bins = compute_bins(_shot(0.0, 60.0), cfg)
        expected_last_end = 60.0 - cfg.boundary_shrink_pct * 60.0
        assert bins[-1][1] == pytest.approx(expected_last_end)


# --------------------------------------------------------------------------- #
# _candidate_times                                                            #
# --------------------------------------------------------------------------- #


class TestCandidateTimes:
    def test_returns_centred_points(self):
        # K = 4 in [0, 1] -> [0.125, 0.375, 0.625, 0.875] (cell midpoints).
        ts = _candidate_times(0.0, 1.0, 4)
        assert ts == pytest.approx([0.125, 0.375, 0.625, 0.875])

    def test_k_one_returns_midpoint(self):
        assert _candidate_times(2.0, 4.0, 1) == [3.0]

    def test_k_zero_returns_empty(self):
        assert _candidate_times(0.0, 1.0, 0) == []


# --------------------------------------------------------------------------- #
# _ordinal_rank                                                               #
# --------------------------------------------------------------------------- #


class TestOrdinalRank:
    def test_empty(self):
        assert _ordinal_rank([]) == []

    def test_single_returns_top(self):
        assert _ordinal_rank([42.0]) == [1.0]

    def test_sorted_ascending(self):
        # 4 elements, ranks evenly spaced 0, 1/3, 2/3, 1.
        assert _ordinal_rank([1.0, 2.0, 3.0, 4.0]) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])

    def test_reversed(self):
        assert _ordinal_rank([4.0, 3.0, 2.0, 1.0]) == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0])

    def test_stable_for_ties(self):
        # Stable sort: equal values keep their input order, so the first
        # occurrence gets the lower rank.
        ranks = _ordinal_rank([5.0, 5.0, 5.0])
        assert ranks[0] < ranks[1] < ranks[2]


# --------------------------------------------------------------------------- #
# score_bin_candidates                                                        #
# --------------------------------------------------------------------------- #


class TestScoreBinCandidates:
    def test_empty(self):
        assert score_bin_candidates([]) == []

    def test_single_full_score_no_saliency(self):
        scores = score_bin_candidates([_qm(saliency_mass=0.0)])
        # 0.6 (rank=1) + 0.2 (rank=1) + 0 = 0.8.
        assert scores == [pytest.approx(0.8)]

    def test_single_with_saliency(self):
        scores = score_bin_candidates([_qm(saliency_mass=0.5)])
        # 0.6 + 0.2 + 0.2 * 0.5 = 0.9.
        assert scores == [pytest.approx(0.9)]

    def test_higher_laplacian_higher_score(self):
        scores = score_bin_candidates(
            [
                _qm(laplacian_var=10.0, entropy=7.0, saliency_mass=0.0),
                _qm(laplacian_var=100.0, entropy=7.0, saliency_mass=0.0),
            ]
        )
        assert scores[1] > scores[0]

    def test_saliency_breaks_a_tie(self):
        scores = score_bin_candidates(
            [
                _qm(laplacian_var=50.0, entropy=7.0, saliency_mass=0.0),
                _qm(laplacian_var=50.0, entropy=7.0, saliency_mass=0.5),
            ]
        )
        assert scores[1] > scores[0]


# --------------------------------------------------------------------------- #
# select_from_bin                                                             #
# --------------------------------------------------------------------------- #


class TestSelectFromBin:
    def test_empty_returns_none(self):
        assert select_from_bin([], QualityGate()) is None

    def test_all_uniform_returns_none(self):
        # Solid colour frames have luma_variance = 0 < 5 -> all rejected.
        cands = [_solid_frame(0.1, gray=128), _solid_frame(0.2, gray=64)]
        assert select_from_bin(cands, QualityGate()) is None

    def test_mixed_picks_a_survivor(self):
        cands = [_solid_frame(0.1, gray=128), _noise_frame(0.2, seed=0)]
        result = select_from_bin(cands, QualityGate())
        assert result is not None
        chosen, metrics, conf = result
        assert chosen.pts.seconds == pytest.approx(0.2)
        assert conf is Confidence.High
        assert metrics.luma_variance > 5.0


# --------------------------------------------------------------------------- #
# extract_for_shot — happy path on varied (high-quality) frames               #
# --------------------------------------------------------------------------- #


class TestExtractForShot:
    def test_noise_video_one_keyframe_per_bin(self, varied_video: Path):
        with VideoDecoder.open(varied_video, target_size=64) as dec:
            shot = ShotRange(
                start=Timestamp(0, dec.timebase),
                end=Timestamp(round(1.4 * dec.timebase.den), dec.timebase),
            )
            keyframes = extract_for_shot(shot, 7, dec, SamplingConfig())
        # D = 1.4s, I = 4s -> N = ceil(1.4/4) = 1.
        assert len(keyframes) == 1
        kf = keyframes[0]
        assert isinstance(kf, ExtractedKeyframe)
        assert kf.shot_id == 7
        assert kf.bucket_index == 0
        assert kf.confidence is Confidence.High
        assert kf.width == 64
        assert kf.height == 64
        assert len(kf.rgb) == 64 * 64 * 3

    def test_returns_n_bins_for_long_shot(self, varied_video: Path):
        cfg = SamplingConfig(target_interval_sec=0.3)
        with VideoDecoder.open(varied_video, target_size=64) as dec:
            shot = ShotRange(
                start=Timestamp(0, dec.timebase),
                end=Timestamp(round(1.4 * dec.timebase.den), dec.timebase),
            )
            keyframes = extract_for_shot(shot, 0, dec, cfg)
        # D = 1.4s, I = 0.3s -> N = ceil(1.4/0.3) = 5.
        assert len(keyframes) == 5
        assert [kf.bucket_index for kf in keyframes] == [0, 1, 2, 3, 4]
        # Selected timestamps strictly increase across bins.
        ts = [kf.timestamp.seconds for kf in keyframes]
        assert ts == sorted(ts)
        assert all(kf.confidence is Confidence.High for kf in keyframes)


# --------------------------------------------------------------------------- #
# extract_for_shot — fallback path on uniform (gate-failing) frames           #
# --------------------------------------------------------------------------- #


class TestExtractForShotFallback:
    def test_uniform_video_yields_degraded(self, tiny_video: Path):
        # Every frame in tiny_video is a single-colour ramp, luma_variance = 0
        # for each, so the gate fails on every probe. Fallback force-picks.
        with VideoDecoder.open(tiny_video) as dec:
            shot = ShotRange(
                start=Timestamp(0, dec.timebase),
                end=Timestamp(round(0.8 * dec.timebase.den), dec.timebase),
            )
            keyframes = extract_for_shot(shot, 0, dec, SamplingConfig())
        assert len(keyframes) == 1
        assert keyframes[0].confidence is Confidence.Degraded


# --------------------------------------------------------------------------- #
# extract_all                                                                 #
# --------------------------------------------------------------------------- #


class TestExtractAll:
    def test_shape_matches_input(self, varied_video: Path):
        with VideoDecoder.open(varied_video, target_size=64) as dec:
            tb = dec.timebase
            shots = [
                ShotRange(start=Timestamp(0, tb), end=Timestamp(round(0.5 * tb.den), tb)),
                ShotRange(
                    start=Timestamp(round(0.6 * tb.den), tb),
                    end=Timestamp(round(1.2 * tb.den), tb),
                ),
            ]
            result = extract_all(shots, dec, SamplingConfig())
        assert len(result) == 2
        # Both shots are < I, so each yields one bin.
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert result[0][0].shot_id == 0
        assert result[1][0].shot_id == 1

    def test_empty_shots_returns_empty(self, varied_video: Path):
        with VideoDecoder.open(varied_video) as dec:
            assert extract_all([], dec, SamplingConfig()) == []
