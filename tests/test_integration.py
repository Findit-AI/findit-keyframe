"""End-to-end tests using a synthetic video.

We generate an 8-second test video on the fly with ffmpeg's ``testsrc``
filter, then run the full extract() pipeline over known shot boundaries.
The video has visible motion, so real decoding + sampling is exercised.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from findit_keyframe import Config, Shot, extract


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None or Path.home().joinpath(
        ".local/bin/ffmpeg"
    ).exists()


def _ffmpeg_binary() -> str:
    local = Path.home() / ".local/bin/ffmpeg"
    return str(local) if local.exists() else "ffmpeg"


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An 8-second 320x240 @ 30 fps testsrc2 video."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")

    out = tmp_path_factory.mktemp("videos") / "testsrc.mp4"
    subprocess.run(
        [
            _ffmpeg_binary(),
            "-y",
            "-f", "lavfi",
            "-i", "testsrc2=size=320x240:rate=30:duration=8",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    assert out.exists()
    return out


def test_extract_single_shot(synthetic_video: Path) -> None:
    shots = [Shot(start_sec=0.0, end_sec=8.0)]
    result = extract(synthetic_video, shots)

    assert len(result) == 1
    keyframes = result[0]
    # 8 s @ target=4 s → 2 buckets, so 2 keyframes.
    assert len(keyframes) == 2
    # All keyframes must land inside the shot.
    for kf in keyframes:
        assert 0.0 <= kf.timestamp_sec <= 8.0
        # 320x240 expected (may differ slightly if container reports fallback).
        assert kf.image.size == (320, 240)
        # testsrc2 is a high-frequency moving pattern → always sharp well
        # above the default min_sharpness (100 at internal 384px scale).
        assert kf.sharpness > 100.0


def test_extract_multiple_shots(synthetic_video: Path) -> None:
    shots = [
        Shot(start_sec=0.0, end_sec=3.0),
        Shot(start_sec=3.0, end_sec=6.0),
        Shot(start_sec=6.0, end_sec=8.0),
    ]
    result = extract(synthetic_video, shots)
    assert len(result) == 3
    # Each short shot → 1 bucket → 1 keyframe.
    for shot_keyframes in result:
        assert len(shot_keyframes) == 1
    # Timestamps should be in the expected shot ranges.
    assert 0.0 <= result[0][0].timestamp_sec < 3.0
    assert 3.0 <= result[1][0].timestamp_sec < 6.0
    assert 6.0 <= result[2][0].timestamp_sec < 8.0


def test_keyframes_within_shot_are_time_sorted(synthetic_video: Path) -> None:
    """A shot's keyframes must be monotonically increasing in timestamp."""
    shots = [Shot(start_sec=0.0, end_sec=8.0)]
    result = extract(synthetic_video, shots)
    timestamps = [kf.timestamp_sec for kf in result[0]]
    assert timestamps == sorted(timestamps)


def test_custom_config_changes_bucket_count(synthetic_video: Path) -> None:
    shots = [Shot(start_sec=0.0, end_sec=8.0)]
    cfg = Config(target_interval_sec=2.0)  # 8/2 = 4 buckets
    result = extract(synthetic_video, shots, config=cfg)
    assert len(result[0]) == 4


def test_pil_image_is_rgb(synthetic_video: Path) -> None:
    shots = [Shot(start_sec=0.0, end_sec=8.0)]
    result = extract(synthetic_video, shots)
    for kf in result[0]:
        assert kf.image.mode == "RGB"


def test_file_not_found_raises() -> None:
    with pytest.raises(FileNotFoundError):
        extract("/nonexistent/video.mp4", [Shot(0.0, 5.0)])


def test_empty_shots_returns_empty_list(synthetic_video: Path) -> None:
    assert extract(synthetic_video, []) == []


def test_shot_beyond_video_returns_empty(synthetic_video: Path) -> None:
    # Video is only 8 s long; shot at 100 s has no content.
    shots = [Shot(start_sec=100.0, end_sec=105.0)]
    result = extract(synthetic_video, shots)
    assert len(result) == 1
    assert result[0] == []
