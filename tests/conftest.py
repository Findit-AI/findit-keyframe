"""Shared pytest fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import av
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _encode_ramp_video(
    path: Path,
    *,
    n_frames: int,
    fps: int,
    width: int,
    height: int,
) -> None:
    """Encode a deterministic linear-gray-ramp video at ``path``.

    Frame ``i`` has nominal gray value ``round(i * 255 / (n_frames - 1))``.
    libx264 + yuv420p is lossy enough to perturb individual pixel values, so
    tests should not assert exact gray equality; the ramp is for ordinal
    checks ("frame 15 is brighter than frame 0").
    """
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        denom = max(n_frames - 1, 1)
        for i in range(n_frames):
            gray = round(i * 255 / denom)
            rgb = np.full((height, width, 3), gray, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 1-second 30-fps 64x64 ramp video; 30 frames total."""
    path = tmp_path_factory.mktemp("videos") / "tiny.mp4"
    _encode_ramp_video(path, n_frames=30, fps=30, width=64, height=64)
    return path


def _encode_textured_video(
    path: Path,
    *,
    n_frames: int,
    fps: int,
    width: int,
    height: int,
) -> None:
    """Encode a video where every frame is mid-tone deterministic noise.

    The luma variance survives the libx264/yuv420p roundtrip cleanly, so the
    sampler tests can exercise the quality gate's *pass* path. Each frame's
    seed is its index, giving frame-to-frame independence and per-frame
    reproducibility.
    """
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "ultrafast"}
        for i in range(n_frames):
            rng = np.random.default_rng(seed=i)
            rgb = rng.integers(50, 201, size=(height, width, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


@pytest.fixture(scope="session")
def varied_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 1.5-second 30-fps 64x64 noise video; high luma variance per frame."""
    path = tmp_path_factory.mktemp("videos") / "varied.mp4"
    _encode_textured_video(path, n_frames=45, fps=30, width=64, height=64)
    return path
