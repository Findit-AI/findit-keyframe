"""argparse-based CLI for findit-keyframe.

Subcommands:

* ``extract VIDEO_PATH SHOTS_JSON OUTPUT_DIR`` — read the shot list, extract
  one keyframe per bin per shot, write each keyframe as a baseline JPEG and
  emit ``manifest.json`` describing all outputs.

Exit codes (per ``TASKS.md`` §7):

* ``0`` — success.
* ``1`` — input error (bad JSON, unknown config field, unsupported flag).
* ``2`` — extraction error (decoder open failure, JPEG write failure, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import av
import numpy as np

from findit_keyframe.decoder import VideoDecoder
from findit_keyframe.sampler import extract_all
from findit_keyframe.types import (
    ExtractedKeyframe,
    SamplingConfig,
    ShotRange,
    Timebase,
    Timestamp,
)

if TYPE_CHECKING:
    from findit_keyframe.saliency import SaliencyProvider

__all__ = ["main"]

EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_EXTRACTION_ERROR = 2


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #


def _parse_shot_json(path: Path) -> list[ShotRange]:
    """Read a scenesdetect-compatible shot JSON file into a ``ShotRange`` list.

    Args:
        path: Filesystem path to a JSON file with a top-level ``shots``
            array. Each entry must have keys ``start_pts``, ``end_pts``,
            ``timebase_num``, ``timebase_den``.

    Returns:
        A list of :class:`ShotRange`, in input order.

    Raises:
        FileNotFoundError: If ``path`` doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If the top-level ``shots`` key or any required entry
            field is missing.
        ValueError: If any shot has ``end <= start`` (per :class:`ShotRange`).
    """
    data: dict[str, Any] = json.loads(path.read_text())
    shots: list[ShotRange] = []
    for entry in data["shots"]:
        timebase = Timebase(num=entry["timebase_num"], den=entry["timebase_den"])
        shots.append(
            ShotRange(
                start=Timestamp(pts=entry["start_pts"], timebase=timebase),
                end=Timestamp(pts=entry["end_pts"], timebase=timebase),
            )
        )
    return shots


def _parse_config_json(path: Path | None) -> SamplingConfig:
    """Apply optional JSON overrides to a default :class:`SamplingConfig`.

    Unknown fields raise :class:`ValueError` so typos surface immediately
    rather than silently no-op-ing.

    Args:
        path: Optional path to a JSON object whose keys are
            :class:`SamplingConfig` field names. ``None`` returns defaults.

    Returns:
        A fresh :class:`SamplingConfig` with overrides applied.

    Raises:
        FileNotFoundError: If ``path`` is given but does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the JSON contains a key that is not a
            :class:`SamplingConfig` field.
    """
    config = SamplingConfig()
    if path is None:
        return config
    overrides: dict[str, Any] = json.loads(path.read_text())
    valid_fields = set(vars(config))
    for name, value in overrides.items():
        if name not in valid_fields:
            raise ValueError(f"Unknown SamplingConfig field: {name!r}")
        setattr(config, name, value)
    return config


# --------------------------------------------------------------------------- #
# JPEG output                                                                 #
# --------------------------------------------------------------------------- #


def _write_jpeg(path: Path, rgb_bytes: bytes, width: int, height: int) -> None:
    """Encode a packed RGB24 byte buffer as a baseline JPEG via PyAV's mjpeg.

    ``yuvj420p`` (full-range YUV) is the standard JPEG sampling; the
    ``image2`` muxer writes a single-frame JPEG file rather than an MJPEG
    container.

    Args:
        path: Output filesystem path. Parent directories must already exist.
        rgb_bytes: Packed RGB24 bytes of length ``width * height * 3``.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Raises:
        av.error.FFmpegError: If FFmpeg cannot write the file (permissions,
            disk full, unsupported pixel format, ...).
        OSError: From the underlying file open.
    """
    rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
    container = av.open(str(path), mode="w", format="image2")
    try:
        stream: Any = container.add_stream("mjpeg")
        stream.pix_fmt = "yuvj420p"
        stream.width = width
        stream.height = height
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _manifest_entry(kf: ExtractedKeyframe, filename: str) -> dict[str, Any]:
    return {
        "shot_id": kf.shot_id,
        "bucket": kf.bucket_index,
        "file": filename,
        "timestamp_sec": kf.timestamp.seconds,
        "quality": asdict(kf.quality),
        "confidence": kf.confidence.value,
    }


def _write_outputs(
    video_path: Path,
    output_dir: Path,
    keyframes_per_shot: list[list[ExtractedKeyframe]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for shot_keyframes in keyframes_per_shot:
        for kf in shot_keyframes:
            filename = f"kf_{kf.shot_id:03d}_{kf.bucket_index:03d}.jpg"
            _write_jpeg(output_dir / filename, kf.rgb, kf.width, kf.height)
            entries.append(_manifest_entry(kf, filename))
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"video": str(video_path), "keyframes": entries}, indent=2))
    return manifest_path


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #


class _Parser(argparse.ArgumentParser):
    """Argparse subclass that exits with code 1 on parse errors (vs. argparse's default 2)."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_INPUT_ERROR, f"error: {message}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="findit-keyframe",
        description="Per-shot keyframe extraction. Consumes scenesdetect output, "
        "writes one JPEG per (shot, bin) plus a manifest.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser(
        "extract",
        help="Extract keyframes for a video given a shot list.",
    )
    extract.add_argument("video", type=Path, help="Source video file.")
    extract.add_argument("shots", type=Path, help="Shot list JSON (scenesdetect-compatible).")
    extract.add_argument("output", type=Path, help="Output directory; created if missing.")
    extract.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional SamplingConfig JSON override file.",
    )
    extract.add_argument(
        "--saliency",
        choices=["none", "apple"],
        default="none",
        help="Saliency provider. 'apple' uses Apple Vision (macOS, requires the .[macos] extra).",
    )
    return parser


# --------------------------------------------------------------------------- #
# Command dispatch                                                            #
# --------------------------------------------------------------------------- #


def _build_saliency_provider(name: str) -> SaliencyProvider | None:
    """Map a ``--saliency`` CLI choice to a provider instance.

    Args:
        name: Choice from the ``--saliency`` flag. Currently ``"none"`` or
            ``"apple"``; argparse ``choices`` enforces this upstream.

    Returns:
        ``None`` for ``"none"`` (the sampler skips the saliency call
        entirely); a fresh :class:`AppleVisionSaliencyProvider` for
        ``"apple"``.

    Raises:
        RuntimeError: If ``"apple"`` is requested off-Darwin or without
            ``pyobjc-framework-Vision`` installed (propagated from the
            provider's constructor).
        ValueError: If ``name`` is not one of the known choices (defensive
            check; argparse normally prevents this).
    """
    if name == "none":
        return None
    if name == "apple":
        from findit_keyframe.saliency import AppleVisionSaliencyProvider

        return AppleVisionSaliencyProvider()
    raise ValueError(f"unknown saliency provider: {name!r}")


def _extract_command(args: argparse.Namespace) -> int:
    """Run the ``extract`` subcommand and return its exit code.

    Args:
        args: Parsed argparse namespace from the ``extract`` subparser.

    Returns:
        ``EXIT_OK`` on success, ``EXIT_INPUT_ERROR`` for bad JSON / unknown
        config field / unsupported saliency, ``EXIT_EXTRACTION_ERROR`` for
        decode/encode failures.
    """
    try:
        shots = _parse_shot_json(args.shots)
        config = _parse_config_json(args.config)
        saliency = _build_saliency_provider(args.saliency)
    except (KeyError, ValueError, RuntimeError, json.JSONDecodeError, FileNotFoundError) as exc:
        print(f"error: invalid input: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        with VideoDecoder.open(args.video, target_size=config.target_size) as decoder:
            keyframes = extract_all(shots, decoder, config, saliency_provider=saliency)
        manifest_path = _write_outputs(args.video, args.output, keyframes)
    except (av.error.FFmpegError, OSError, ValueError, RuntimeError) as exc:
        # Programmer bugs (TypeError, AttributeError, etc.) deliberately
        # propagate as crashes so they surface in tracebacks instead of
        # being mapped to the extraction-error exit code.
        print(f"error: extraction failed: {exc}", file=sys.stderr)
        return EXIT_EXTRACTION_ERROR

    n_keyframes = sum(len(s) for s in keyframes)
    print(f"wrote {n_keyframes} keyframes to {args.output}")
    print(f"manifest: {manifest_path}")
    return EXIT_OK


def main() -> int:
    """Console-script entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "extract":
        return _extract_command(args)
    parser.print_help()
    return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
