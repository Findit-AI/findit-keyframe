"""PyAV-backed video decoder with two strategies.

* ``decode_at(time)`` — keyframe seek + forward decode to the target PTS.
  Cheap when shots are sparse; pays a seek penalty per call.
* ``decode_sequential(shots)`` — single linear pass through the file,
  yielding frames whose PTS lies inside any provided shot range. Cheap when
  shots are dense; pays no seek penalty but reads the whole file once.

``pick_strategy`` chooses between them based on shot density and count.
The thresholds are documented in ``TASKS.md`` §3 (density > 0.3 shots/s
or count > 200 → Sequential).

The Rust port (``ffmpeg-next``) preserves the same public surface.
See ``docs/rust-porting.md`` §2.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import av

from findit_keyframe.types import ShotRange, Timebase, Timestamp

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import TracebackType

    import numpy as np
    import numpy.typing as npt


__all__ = [
    "DecodedFrame",
    "Strategy",
    "VideoDecoder",
    "pick_strategy",
]


class Strategy(enum.StrEnum):
    """Frame-fetch strategy for a video + shot list."""

    Sequential = "sequential"
    PerShotSeek = "per_shot_seek"


@dataclass(frozen=True, slots=False)
class DecodedFrame:
    """A decoded RGB24 frame with its presentation timestamp.

    ``rgb`` has shape ``(height, width, 3)`` and dtype ``uint8``. ``width``
    and ``height`` are stored explicitly to mirror the Rust struct (where
    ``rgb`` becomes a ``Vec<u8>`` carrying no shape info).
    """

    pts: Timestamp
    width: int
    height: int
    rgb: npt.NDArray[np.uint8] = field(repr=False)

    def __post_init__(self) -> None:
        expected = (self.height, self.width, 3)
        if self.rgb.shape != expected:
            raise ValueError(
                f"rgb shape {self.rgb.shape} does not match (height, width, 3) = {expected}"
            )


def pick_strategy(shots: list[ShotRange], duration_sec: float) -> Strategy:
    """Choose ``Sequential`` for dense or numerous shots, ``PerShotSeek`` otherwise.

    Returns ``PerShotSeek`` for empty shot lists and unknown durations
    (``duration_sec <= 0``) so callers don't have to special-case those paths.
    """
    if not shots or duration_sec <= 0.0:
        return Strategy.PerShotSeek
    density = len(shots) / duration_sec
    if density > 0.3 or len(shots) > 200:
        return Strategy.Sequential
    return Strategy.PerShotSeek


class VideoDecoder:
    """PyAV-backed video decoder.

    Construct via ``VideoDecoder.open(path, target_size=...)``. Use as a
    context manager (or call ``close()`` explicitly) to release the
    underlying container.

    ``target_size`` (in pixels) controls the output frame's square edge
    length; a value of ``0`` keeps the native resolution.
    """

    def __init__(
        self,
        container: Any,
        stream: Any,
        target_size: int = 0,
    ) -> None:
        self._container = container
        self._stream = stream
        self._target_size = target_size

        tb = stream.time_base
        self._timebase = Timebase(num=tb.numerator, den=tb.denominator)

        if stream.duration is not None:
            self._duration_sec = float(stream.duration * tb)
        else:
            self._duration_sec = 0.0

        self._fps = float(stream.average_rate) if stream.average_rate else 0.0
        self._native_width = int(stream.codec_context.width)
        self._native_height = int(stream.codec_context.height)

    @classmethod
    def open(cls, path: Path, target_size: int = 0) -> Self:
        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            return cls(container, stream, target_size=target_size)
        except Exception:
            container.close()
            raise

    @property
    def duration_sec(self) -> float:
        return self._duration_sec

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def timebase(self) -> Timebase:
        return self._timebase

    @property
    def width(self) -> int:
        return self._target_size if self._target_size else self._native_width

    @property
    def height(self) -> int:
        return self._target_size if self._target_size else self._native_height

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Frame conversion                                                   #
    # ------------------------------------------------------------------ #

    def _to_decoded(self, frame: Any) -> DecodedFrame:
        if self._target_size:
            frame = frame.reformat(
                width=self._target_size,
                height=self._target_size,
                format="rgb24",
            )
        else:
            frame = frame.reformat(format="rgb24")
        rgb: npt.NDArray[np.uint8] = frame.to_ndarray()
        return DecodedFrame(
            pts=Timestamp(int(frame.pts), self._timebase),
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            rgb=rgb,
        )

    # ------------------------------------------------------------------ #
    # decode_at                                                          #
    # ------------------------------------------------------------------ #

    def decode_at(self, time_sec: float) -> DecodedFrame:
        """Seek + decode the first frame whose PTS is at or after ``time_sec``.

        Implementation: seek backward to the keyframe at-or-before the target,
        then decode forward until the first frame with ``pts >= target_pts``.
        Resolves to within ±1 frame of the requested time.
        """
        target_pts = round(time_sec * self._timebase.den / self._timebase.num)
        self._container.seek(target_pts, stream=self._stream, any_frame=False)
        for frame in self._container.decode(self._stream):
            if frame.pts is None:
                continue
            if frame.pts >= target_pts:
                return self._to_decoded(frame)
        raise ValueError(f"Could not decode any frame at or after {time_sec} s")

    # ------------------------------------------------------------------ #
    # decode_sequential                                                  #
    # ------------------------------------------------------------------ #

    def decode_sequential(
        self,
        shots: list[ShotRange],
    ) -> Iterator[tuple[int, DecodedFrame]]:
        """Single-pass scan; yield ``(shot_id, frame)`` for frames inside any shot.

        ``shot_id`` is the original index into ``shots`` (preserved when the
        input list is unsorted). Shots are assumed non-overlapping; behaviour
        with overlapping ranges is unspecified.
        """
        if not shots:
            return

        sorted_shots = sorted(enumerate(shots), key=lambda item: item[1].start)
        cursor = 0

        self._container.seek(0, stream=self._stream)
        for frame in self._container.decode(self._stream):
            if frame.pts is None:
                continue
            ts = Timestamp(int(frame.pts), self._timebase)

            while cursor < len(sorted_shots) and sorted_shots[cursor][1].end <= ts:
                cursor += 1
            if cursor >= len(sorted_shots):
                return

            shot_id, shot = sorted_shots[cursor]
            if shot.start <= ts < shot.end:
                yield shot_id, self._to_decoded(frame)
