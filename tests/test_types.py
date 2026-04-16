"""Tests for ``findit_keyframe.types``.

These tests pin down invariants documented in ``TASKS.md`` §2 and the Type Map
in ``docs/rust-porting.md``. They are the contract the Rust port must replay.
"""

from __future__ import annotations

import dataclasses
from typing import get_type_hints

import pytest

from findit_keyframe.types import (
    Confidence,
    ExtractedKeyframe,
    QualityMetrics,
    SamplingConfig,
    ShotRange,
    Timebase,
    Timestamp,
)

# --------------------------------------------------------------------------- #
# Timebase                                                                    #
# --------------------------------------------------------------------------- #


class TestTimebase:
    def test_zero_denominator_rejected(self):
        with pytest.raises(ValueError, match="den"):
            Timebase(num=1, den=0)

    def test_negative_denominator_rejected(self):
        with pytest.raises(ValueError, match="den"):
            Timebase(num=1, den=-1)

    def test_zero_numerator_accepted(self):
        # scenesdetect allows num == 0; mirror that.
        tb = Timebase(num=0, den=1)
        assert tb.num == 0
        assert tb.den == 1

    def test_value_equality_across_reduced_forms(self):
        # 1/2 == 2/4 == 5/10 — semantic equality, not field equality.
        assert Timebase(1, 2) == Timebase(2, 4)
        assert Timebase(1, 2) == Timebase(5, 10)
        assert Timebase(1, 1000) == Timebase(90, 90000)

    def test_inequality_across_reduced_forms(self):
        assert Timebase(1, 2) != Timebase(1, 3)
        assert Timebase(2, 5) != Timebase(3, 5)

    def test_hash_matches_equality(self):
        assert hash(Timebase(1, 2)) == hash(Timebase(2, 4))
        assert hash(Timebase(1, 1000)) == hash(Timebase(90, 90000))
        # And distinct values usually have distinct hashes (sanity).
        assert hash(Timebase(1, 2)) != hash(Timebase(1, 3))

    def test_is_frozen(self):
        tb = Timebase(1, 1000)
        with pytest.raises(dataclasses.FrozenInstanceError):
            tb.num = 2  # type: ignore[misc]

    def test_repr_readable(self):
        assert "1/1000" in repr(Timebase(1, 1000))


# --------------------------------------------------------------------------- #
# Timestamp                                                                   #
# --------------------------------------------------------------------------- #


class TestTimestamp:
    def test_seconds_simple_case(self):
        # 1000 ticks at 1/1000 timebase == 1.0 second.
        assert Timestamp(pts=1000, timebase=Timebase(1, 1000)).seconds == pytest.approx(1.0)

    def test_seconds_video_timebase(self):
        # 90000 ticks at 1/90000 (MPEG-TS) == 1.0 second.
        assert Timestamp(pts=90000, timebase=Timebase(1, 90000)).seconds == pytest.approx(1.0)

    def test_seconds_zero(self):
        assert Timestamp(pts=0, timebase=Timebase(1, 1000)).seconds == 0.0

    def test_cross_timebase_equality(self):
        a = Timestamp(pts=1000, timebase=Timebase(1, 1000))
        b = Timestamp(pts=90000, timebase=Timebase(1, 90000))
        assert a == b
        assert hash(a) == hash(b)

    def test_ordering_same_timebase(self):
        tb = Timebase(1, 1000)
        assert Timestamp(500, tb) < Timestamp(1000, tb)
        assert Timestamp(1000, tb) > Timestamp(500, tb)
        assert Timestamp(500, tb) <= Timestamp(500, tb)

    def test_ordering_cross_timebase(self):
        a = Timestamp(pts=500, timebase=Timebase(1, 1000))  # 0.5s
        b = Timestamp(pts=90000, timebase=Timebase(1, 90000))  # 1.0s
        assert a < b
        assert b > a

    def test_is_frozen(self):
        ts = Timestamp(0, Timebase(1, 1000))
        with pytest.raises(dataclasses.FrozenInstanceError):
            ts.pts = 1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# ShotRange                                                                   #
# --------------------------------------------------------------------------- #


class TestShotRange:
    def test_duration_sec_basic(self):
        tb = Timebase(1, 1000)
        sr = ShotRange(start=Timestamp(0, tb), end=Timestamp(5000, tb))
        assert sr.duration_sec == pytest.approx(5.0)

    def test_duration_sec_cross_timebase(self):
        sr = ShotRange(
            start=Timestamp(0, Timebase(1, 1000)),
            end=Timestamp(90000, Timebase(1, 90000)),
        )
        assert sr.duration_sec == pytest.approx(1.0)

    def test_end_before_start_rejected(self):
        tb = Timebase(1, 1000)
        with pytest.raises(ValueError, match="end"):
            ShotRange(start=Timestamp(1000, tb), end=Timestamp(500, tb))

    def test_zero_duration_rejected(self):
        tb = Timebase(1, 1000)
        with pytest.raises(ValueError, match="end"):
            ShotRange(start=Timestamp(1000, tb), end=Timestamp(1000, tb))

    def test_is_frozen(self):
        tb = Timebase(1, 1000)
        sr = ShotRange(Timestamp(0, tb), Timestamp(1000, tb))
        with pytest.raises(dataclasses.FrozenInstanceError):
            sr.start = Timestamp(1, tb)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# SamplingConfig                                                              #
# --------------------------------------------------------------------------- #


class TestSamplingConfig:
    def test_defaults_match_spec(self):
        c = SamplingConfig()
        assert c.target_interval_sec == pytest.approx(4.0)
        assert c.candidates_per_bin == 6
        assert c.max_frames_per_shot == 16
        assert c.boundary_shrink_pct == pytest.approx(0.02)
        assert c.fallback_expand_pct == pytest.approx(0.20)
        assert c.target_size == 384

    def test_is_mutable(self):
        # SamplingConfig is one of two intentionally non-frozen dataclasses.
        c = SamplingConfig()
        c.target_size = 256
        assert c.target_size == 256

    def test_replace_returns_new_instance(self):
        c1 = SamplingConfig()
        c2 = dataclasses.replace(c1, target_size=256)
        assert c2.target_size == 256
        assert c1.target_size == 384


# --------------------------------------------------------------------------- #
# Confidence                                                                  #
# --------------------------------------------------------------------------- #


class TestConfidence:
    def test_three_levels(self):
        assert {Confidence.High, Confidence.Low, Confidence.Degraded} == set(Confidence)

    def test_string_value_lowercase(self):
        # The CLI manifest uses lowercase string form.
        assert Confidence.High.value == "high"
        assert Confidence.Low.value == "low"
        assert Confidence.Degraded.value == "degraded"


# --------------------------------------------------------------------------- #
# QualityMetrics                                                              #
# --------------------------------------------------------------------------- #


class TestQualityMetrics:
    def test_construction_and_fields(self):
        q = QualityMetrics(
            laplacian_var=215.4,
            mean_luma=0.41,
            luma_variance=1820.7,
            entropy=7.31,
            saliency_mass=0.62,
        )
        assert q.laplacian_var == pytest.approx(215.4)
        assert q.mean_luma == pytest.approx(0.41)
        assert q.luma_variance == pytest.approx(1820.7)
        assert q.entropy == pytest.approx(7.31)
        assert q.saliency_mass == pytest.approx(0.62)

    def test_is_frozen(self):
        q = QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            q.entropy = 1.0  # type: ignore[misc]

    def test_all_fields_are_floats(self):
        hints = get_type_hints(QualityMetrics)
        assert hints["laplacian_var"] is float
        assert hints["mean_luma"] is float
        assert hints["luma_variance"] is float
        assert hints["entropy"] is float
        assert hints["saliency_mass"] is float


# --------------------------------------------------------------------------- #
# ExtractedKeyframe                                                           #
# --------------------------------------------------------------------------- #


class TestExtractedKeyframe:
    def _make(self, **overrides):
        defaults = {
            "shot_id": 0,
            "timestamp": Timestamp(1000, Timebase(1, 1000)),
            "bucket_index": 0,
            "rgb": b"\x00" * (4 * 4 * 3),
            "width": 4,
            "height": 4,
            "quality": QualityMetrics(0.0, 0.5, 1.0, 7.0, 0.0),
            "confidence": Confidence.High,
        }
        return ExtractedKeyframe(**(defaults | overrides))

    def test_construction(self):
        kf = self._make()
        assert kf.shot_id == 0
        assert kf.bucket_index == 0
        assert kf.width == 4
        assert kf.height == 4
        assert kf.confidence is Confidence.High
        assert kf.quality.entropy == pytest.approx(7.0)

    def test_is_mutable(self):
        # Per TASKS.md §2 — ExtractedKeyframe is intentionally mutable so callers
        # can attach downstream metadata before serialising.
        kf = self._make()
        kf.confidence = Confidence.Low
        assert kf.confidence is Confidence.Low

    def test_rgb_is_bytes(self):
        kf = self._make()
        assert isinstance(kf.rgb, bytes)
        assert len(kf.rgb) == kf.width * kf.height * 3
