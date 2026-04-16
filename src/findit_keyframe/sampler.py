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

from findit_keyframe.quality import QualityGate, compute_quality
from findit_keyframe.types import (
    Confidence,
    ExtractedKeyframe,
    QualityMetrics,
    SamplingConfig,
    ShotRange,
)

if TYPE_CHECKING:
    from findit_keyframe.decoder import DecodedFrame, VideoDecoder
    from findit_keyframe.saliency import SaliencyProvider


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

    Args:
        shot: The shot to partition. ``shot.duration_sec > 0`` is required
            by :class:`ShotRange`'s constructor; this function additionally
            handles a defensive ``<= 0`` case by returning an empty list.
        config: Sampling parameters; ``target_interval_sec``,
            ``max_frames_per_shot``, and ``boundary_shrink_pct`` are read.

    Returns:
        A list of ``(t0, t1)`` second-valued tuples, all of equal width
        and contiguous (``bins[i].t1 == bins[i+1].t0``).
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
    """Compute the composite quality score for each candidate in a bin.

    Formula: ``score = 0.6 * rank(laplacian_var) + 0.2 * rank(entropy) +
    0.2 * saliency_mass``. ``rank`` is the stable ordinal rank within the
    bin's pool, scaled to ``[0, 1]``. Single-candidate bins use ``rank = 1``
    for both quality terms, so the score collapses to ``0.8 + 0.2 *
    saliency_mass``.

    Args:
        metrics_list: Per-candidate quality metrics, in candidate order.

    Returns:
        A parallel list of float scores in candidate order. Empty input
        returns an empty list.
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


def _compute_metrics(
    candidates: list[DecodedFrame],
    saliency_provider: SaliencyProvider | None,
) -> list[QualityMetrics]:
    if saliency_provider is None:
        return [compute_quality(c.rgb) for c in candidates]
    return [compute_quality(c.rgb, saliency=saliency_provider.compute(c.rgb)) for c in candidates]


def select_from_bin(
    candidates: list[DecodedFrame],
    quality_gate: QualityGate,
    saliency_provider: SaliencyProvider | None = None,
) -> tuple[DecodedFrame, QualityMetrics, Confidence] | None:
    """Apply the gate, score survivors, return the highest-scoring one.

    Args:
        candidates: Decoded candidate frames for this bin. Empty input
            returns ``None``.
        quality_gate: Hard pass/fail predicate applied to each candidate.
        saliency_provider: Optional saliency contributor to the composite
            score; ``None`` skips the per-frame saliency call entirely.

    Returns:
        ``(frame, metrics, Confidence.High)`` for the winning candidate, or
        ``None`` when every candidate fails the gate. The fallback path
        wraps the confidence to ``Low`` or ``Degraded`` as appropriate.
    """
    if not candidates:
        return None
    metrics = _compute_metrics(candidates, saliency_provider)
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
    saliency_provider: SaliencyProvider | None = None,
) -> tuple[DecodedFrame, QualityMetrics, Confidence]:
    """Try the native bin, then an expanded window, then force-pick.

    Always returns a frame; the caller never has to handle ``None``.

    Args:
        bin_idx: Index of the bin to fill, in ``[0, len(bins))``.
        bins: Output of :func:`compute_bins` for the same shot.
        shot: The parent shot; used to clamp the expanded window.
        decoder: Decoder for fetching candidate frames.
        config: Sampling parameters (probe count, fallback expansion, ...).
        quality_gate: Hard pass/fail predicate.
        saliency_provider: Optional saliency contributor; ``None`` skips it.

    Returns:
        ``(frame, metrics, confidence)`` for the bin's selected keyframe.
        The confidence ladder is ``High`` (native pool) → ``Low`` (expanded
        pool) → ``Degraded`` (force-picked from a gate-failing pool).
    """
    t0, t1 = bins[bin_idx]
    bin_width = t1 - t0

    native = [decoder.decode_at(t) for t in _candidate_times(t0, t1, config.candidates_per_bin)]
    native_pick = select_from_bin(native, quality_gate, saliency_provider)
    if native_pick is not None:
        return native_pick

    expand = config.fallback_expand_pct * bin_width
    et0 = max(shot.start.seconds, t0 - expand)
    et1 = min(shot.end.seconds, t1 + expand)
    expanded = [decoder.decode_at(t) for t in _candidate_times(et0, et1, config.candidates_per_bin)]
    expanded_pick = select_from_bin(expanded, quality_gate, saliency_provider)
    if expanded_pick is not None:
        frame, metrics, _ = expanded_pick
        return frame, metrics, Confidence.Low

    pool = native + expanded
    metrics_pool = _compute_metrics(pool, saliency_provider)
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
    saliency_provider: SaliencyProvider | None = None,
) -> list[ExtractedKeyframe]:
    """Extract one keyframe per bin for a single shot.

    Args:
        shot: The shot to process.
        shot_id: Identifier copied verbatim into every emitted
            :class:`ExtractedKeyframe`. The caller chooses the numbering
            scheme; :func:`extract_all` uses input-list index.
        decoder: An open :class:`VideoDecoder` covering ``shot``.
        config: Sampling parameters.
        quality_gate: Optional override; defaults to :class:`QualityGate()`.
        saliency_provider: Optional saliency contributor; ``None`` means
            ``saliency_mass = 0``.

    Returns:
        Exactly ``len(compute_bins(shot, config))`` keyframes (one per bin),
        each tagged with the bin's ``bucket_index`` and a
        :class:`Confidence`. The ``rgb`` buffer is materialised as ``bytes``
        so the result is safely serialisable.

    Raises:
        ValueError: Propagated from :meth:`VideoDecoder.decode_at` when a
            probe falls past the end of the stream (typically a malformed
            shot whose end exceeds the decoder's duration).
    """
    bins = compute_bins(shot, config)
    gate = quality_gate or QualityGate()
    out: list[ExtractedKeyframe] = []
    for i in range(len(bins)):
        frame, metrics, confidence = _select_with_fallback(
            i, bins, shot, decoder, config, gate, saliency_provider
        )
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
    saliency_provider: SaliencyProvider | None = None,
) -> list[list[ExtractedKeyframe]]:
    """Extract keyframes for every shot in ``shots``.

    Per-shot extraction goes through :func:`extract_for_shot`; this function
    is only an iteration shell. The decoder's per-shot-seek path is the only
    one currently implemented — see :class:`findit_keyframe.decoder.Strategy`
    and :func:`findit_keyframe.decoder.pick_strategy` for the dense-shot
    Sequential optimisation tracked for the Rust port.

    Args:
        shots: Input shot list, in any order. The output preserves input order.
        decoder: An open :class:`VideoDecoder` covering the same video.
        config: Sampling parameters applied to every shot.
        quality_gate: Optional override; defaults to :class:`QualityGate()`.
        saliency_provider: Optional saliency contributor; ``None`` means
            ``saliency_mass = 0`` for every keyframe.

    Returns:
        A list with one entry per input shot, each entry a list of
        :class:`ExtractedKeyframe` (one per bin in that shot).
    """
    return [
        extract_for_shot(shot, shot_id, decoder, config, quality_gate, saliency_provider)
        for shot_id, shot in enumerate(shots)
    ]
