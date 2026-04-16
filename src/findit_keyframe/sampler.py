"""Stratified temporal sampler with quality-gated within-bin selection.

The algorithm is documented in ``docs/algorithm.md``; this module is the
reference implementation. Each function maps 1:1 to a Rust function — see
``docs/rust-porting.md`` §3 for the idiom map.

High-level flow per shot:

1. ``compute_bins`` — partition the (boundary-shrunken) shot into N bins.
2. For each bin, probe K cell-centred candidate timestamps.
3. ``select_from_bin`` — apply ``QualityGate``, score survivors,
   ``argmax`` -> emit ``Confidence.High``.
4. On bin failure, expand ±``fallback_expand_pct`` of bin width into
   neighbours; retry. Surviving pick is tagged ``Confidence.Low``.
5. On expansion failure, force-pick the highest-score candidate even
   though it failed the gate; emit ``Confidence.Degraded``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from findit_keyframe.decoder import VideoDecoder, pick_strategy
from findit_keyframe.quality import QualityGate, compute_quality
from findit_keyframe.types import (
    Confidence,
    ExtractedKeyframe,
    QualityMetrics,
    SamplingConfig,
    ShotRange,
)

if TYPE_CHECKING:
    from findit_keyframe.decoder import DecodedFrame


__all__ = [
    "compute_bins",
    "extract_all",
    "extract_for_shot",
    "score_bin_candidates",
    "select_from_bin",
]


# --------------------------------------------------------------------------- #
# Binning                                                                     #
# --------------------------------------------------------------------------- #


def compute_bins(shot: ShotRange, config: SamplingConfig) -> list[tuple[float, float]]:
    """Partition ``shot`` into N equal bins after symmetric boundary shrinkage.

    ``N = clamp(ceil(D / target_interval_sec), 1, max_frames_per_shot)`` where
    ``D = shot.duration_sec``. The shrunken effective range is
    ``[start + s, end - s]`` with ``s = boundary_shrink_pct * D``; bins divide
    this range evenly and are returned as ``(t0, t1)`` half-open intervals.
    """
    duration = shot.duration_sec
    if duration <= 0.0:
        return []
    n = max(
        1,
        min(config.max_frames_per_shot, math.ceil(duration / config.target_interval_sec)),
    )
    shrink = config.boundary_shrink_pct * duration
    start = shot.start.seconds + shrink
    end = shot.end.seconds - shrink
    width = (end - start) / n
    return [(start + i * width, start + (i + 1) * width) for i in range(n)]


# --------------------------------------------------------------------------- #
# Candidate sampling                                                          #
# --------------------------------------------------------------------------- #


def _candidate_times(t0: float, t1: float, k: int) -> list[float]:
    """``k`` evenly-spaced cell-centre timestamps in ``[t0, t1]``.

    Cell centres (rather than endpoints) avoid sampling exactly on bin
    boundaries, which is where the upstream cut detector is least confident.
    """
    if k <= 0:
        return []
    if k == 1:
        return [(t0 + t1) / 2.0]
    width = (t1 - t0) / k
    return [t0 + (i + 0.5) * width for i in range(k)]


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #


def _ordinal_rank(values: list[float]) -> list[float]:
    """Stable ordinal rank in ``[0, 1]`` (lowest -> 0, highest -> 1).

    Ties keep input order: the earlier-seen value gets the lower rank. This
    is sufficient because ``argmax`` on the composite score is deterministic
    either way and the algorithm does not depend on tie-broken averaging.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = np.asarray(values, dtype=np.float64).argsort(kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n) / (n - 1)
    return [float(r) for r in ranks]


def score_bin_candidates(metrics_list: list[QualityMetrics]) -> list[float]:
    """Composite quality score for each candidate in a bin.

    ``score = 0.6 * rank(laplacian_var) + 0.2 * rank(entropy) + 0.2 * saliency_mass``.
    Single-candidate bins return the maximum possible quality contribution
    (rank = 1) plus their saliency.
    """
    if not metrics_list:
        return []
    if len(metrics_list) == 1:
        return [0.6 + 0.2 + 0.2 * metrics_list[0].saliency_mass]
    lap = _ordinal_rank([m.laplacian_var for m in metrics_list])
    ent = _ordinal_rank([m.entropy for m in metrics_list])
    return [
        0.6 * lap[i] + 0.2 * ent[i] + 0.2 * metrics_list[i].saliency_mass
        for i in range(len(metrics_list))
    ]


# --------------------------------------------------------------------------- #
# Selection                                                                   #
# --------------------------------------------------------------------------- #


def select_from_bin(
    candidates: list[DecodedFrame],
    quality_gate: QualityGate,
) -> tuple[DecodedFrame, QualityMetrics, Confidence] | None:
    """Apply the gate, score survivors, return the highest-scoring one.

    Returns ``None`` when every candidate fails the hard gate. The returned
    ``Confidence`` is always ``High`` here; the fallback path wraps it to
    ``Low`` or ``Degraded`` as appropriate.
    """
    if not candidates:
        return None
    metrics = [compute_quality(c.rgb) for c in candidates]
    survivors = [(f, m) for f, m in zip(candidates, metrics, strict=True) if quality_gate.passes(m)]
    if not survivors:
        return None
    surviving_metrics = [m for _, m in survivors]
    scores = score_bin_candidates(surviving_metrics)
    best = int(np.argmax(scores))
    frame, mtr = survivors[best]
    return frame, mtr, Confidence.High


def _select_with_fallback(
    bin_idx: int,
    bins: list[tuple[float, float]],
    shot: ShotRange,
    decoder: VideoDecoder,
    config: SamplingConfig,
    quality_gate: QualityGate,
) -> tuple[DecodedFrame, QualityMetrics, Confidence]:
    """Native bin -> expanded window -> force-pick. Always returns a frame."""
    t0, t1 = bins[bin_idx]
    bin_width = t1 - t0

    native = [decoder.decode_at(t) for t in _candidate_times(t0, t1, config.candidates_per_bin)]
    if (result := select_from_bin(native, quality_gate)) is not None:
        return result

    expand = config.fallback_expand_pct * bin_width
    et0 = max(shot.start.seconds, t0 - expand)
    et1 = min(shot.end.seconds, t1 + expand)
    expanded = [decoder.decode_at(t) for t in _candidate_times(et0, et1, config.candidates_per_bin)]
    if (result := select_from_bin(expanded, quality_gate)) is not None:
        frame, metrics, _ = result
        return frame, metrics, Confidence.Low

    pool = native + expanded
    metrics_pool = [compute_quality(c.rgb) for c in pool]
    scores = score_bin_candidates(metrics_pool)
    best = int(np.argmax(scores))
    return pool[best], metrics_pool[best], Confidence.Degraded


# --------------------------------------------------------------------------- #
# Top-level entry points                                                      #
# --------------------------------------------------------------------------- #


def extract_for_shot(
    shot: ShotRange,
    shot_id: int,
    decoder: VideoDecoder,
    config: SamplingConfig,
    quality_gate: QualityGate | None = None,
) -> list[ExtractedKeyframe]:
    """Extract one keyframe per bin for a single shot."""
    bins = compute_bins(shot, config)
    gate = quality_gate or QualityGate()
    out: list[ExtractedKeyframe] = []
    for i in range(len(bins)):
        frame, metrics, confidence = _select_with_fallback(i, bins, shot, decoder, config, gate)
        out.append(
            ExtractedKeyframe(
                shot_id=shot_id,
                timestamp=frame.pts,
                bucket_index=i,
                rgb=frame.rgb.tobytes(),
                width=frame.width,
                height=frame.height,
                quality=metrics,
                confidence=confidence,
            )
        )
    return out


def extract_all(
    shots: list[ShotRange],
    decoder: VideoDecoder,
    config: SamplingConfig,
    quality_gate: QualityGate | None = None,
) -> list[list[ExtractedKeyframe]]:
    """Extract keyframes for every shot in ``shots``.

    The return shape mirrors the input: one ``list[ExtractedKeyframe]`` per
    shot, in input order. ``pick_strategy`` is consulted for telemetry but
    P2 always uses the per-shot seek path; the dense-shot Sequential
    optimisation lands in P3.
    """
    _ = pick_strategy(shots, decoder.duration_sec)
    return [
        extract_for_shot(shot, shot_id, decoder, config, quality_gate)
        for shot_id, shot in enumerate(shots)
    ]
