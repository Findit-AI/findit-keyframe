"""findit-keyframe: temporally-distributed, quality-aware keyframes for VLM pipelines.

Public API::

    from findit_keyframe import extract, Shot, Config

    shots = [Shot(start_sec=0.0, end_sec=12.3), Shot(start_sec=12.3, end_sec=47.8)]
    keyframes_per_shot = extract("video.mp4", shots)

    for kf in keyframes_per_shot[0]:
        kf.image.save(f"{kf.timestamp_sec:.2f}.jpg")
"""

from .extractor import extract
from .types import Config, Keyframe, Shot

__version__ = "0.1.0"
__all__ = ["Config", "Keyframe", "Shot", "extract"]
