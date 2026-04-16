"""Tests for ``findit_keyframe.cli``.

Covers:

* JSON parsers for shot lists and config overrides.
* The ``extract`` subcommand end-to-end on the ``varied_video`` fixture
  (manifest contents, JPEG validity via PyAV round-trip).
* Exit codes per ``TASKS.md`` §7: 0 success, 1 input error, 2 extraction error.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import av
import pytest

from findit_keyframe.cli import _parse_config_json, _parse_shot_json, main

if TYPE_CHECKING:
    from pathlib import Path


def _write_shots_json(path: Path, shots: list[dict]) -> None:
    path.write_text(json.dumps({"shots": shots}))


# --------------------------------------------------------------------------- #
# _parse_shot_json                                                            #
# --------------------------------------------------------------------------- #


class TestParseShotJson:
    def test_basic(self, tmp_path: Path):
        p = tmp_path / "shots.json"
        _write_shots_json(
            p,
            [
                {"id": 0, "start_pts": 0, "end_pts": 1000, "timebase_num": 1, "timebase_den": 1000},
                {
                    "id": 1,
                    "start_pts": 1000,
                    "end_pts": 2000,
                    "timebase_num": 1,
                    "timebase_den": 1000,
                },
            ],
        )
        shots = _parse_shot_json(p)
        assert len(shots) == 2
        assert shots[0].start.seconds == 0.0
        assert shots[0].end.seconds == 1.0
        assert shots[1].start.seconds == 1.0

    def test_supports_video_timebase(self, tmp_path: Path):
        p = tmp_path / "shots.json"
        _write_shots_json(
            p,
            [{"id": 0, "start_pts": 0, "end_pts": 90000, "timebase_num": 1, "timebase_den": 90000}],
        )
        shots = _parse_shot_json(p)
        assert shots[0].end.seconds == 1.0

    def test_missing_shots_key_raises(self, tmp_path: Path):
        p = tmp_path / "shots.json"
        p.write_text(json.dumps({"other": []}))
        with pytest.raises(KeyError):
            _parse_shot_json(p)

    def test_invalid_shot_range_raises(self, tmp_path: Path):
        # end <= start violates ShotRange invariant.
        p = tmp_path / "shots.json"
        _write_shots_json(
            p,
            [{"id": 0, "start_pts": 1000, "end_pts": 500, "timebase_num": 1, "timebase_den": 1000}],
        )
        with pytest.raises(ValueError, match="end"):
            _parse_shot_json(p)


# --------------------------------------------------------------------------- #
# _parse_config_json                                                          #
# --------------------------------------------------------------------------- #


class TestParseConfigJson:
    def test_none_returns_defaults(self):
        c = _parse_config_json(None)
        assert c.target_size == 384
        assert c.candidates_per_bin == 6

    def test_overrides_applied(self, tmp_path: Path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"target_size": 256, "candidates_per_bin": 8}))
        c = _parse_config_json(p)
        assert c.target_size == 256
        assert c.candidates_per_bin == 8
        # Unspecified fields keep defaults.
        assert c.max_frames_per_shot == 16

    def test_unknown_field_raises(self, tmp_path: Path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"foo": 1}))
        with pytest.raises(ValueError, match="Unknown"):
            _parse_config_json(p)


# --------------------------------------------------------------------------- #
# main: --help                                                                #
# --------------------------------------------------------------------------- #


class TestCliHelp:
    def test_help_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        monkeypatch.setattr(sys, "argv", ["findit-keyframe", "--help"])
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "extract" in out


# --------------------------------------------------------------------------- #
# main: extract — end-to-end                                                  #
# --------------------------------------------------------------------------- #


class TestCliExtract:
    @staticmethod
    def _setup(tmp_path: Path) -> tuple[Path, Path]:
        shots_path = tmp_path / "shots.json"
        _write_shots_json(
            shots_path,
            [
                {"id": 0, "start_pts": 0, "end_pts": 500, "timebase_num": 1, "timebase_den": 1000},
                {
                    "id": 1,
                    "start_pts": 600,
                    "end_pts": 1200,
                    "timebase_num": 1,
                    "timebase_den": 1000,
                },
            ],
        )
        return shots_path, tmp_path / "out"

    def test_writes_jpegs_and_manifest(
        self, tmp_path: Path, varied_video: Path, monkeypatch: pytest.MonkeyPatch
    ):
        shots_path, out_dir = self._setup(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "findit-keyframe",
                "extract",
                str(varied_video),
                str(shots_path),
                str(out_dir),
            ],
        )
        rc = main()
        assert rc == 0

        manifest_path = out_dir / "manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["video"] == str(varied_video)
        assert len(manifest["keyframes"]) >= 2

        for entry in manifest["keyframes"]:
            assert {
                "shot_id",
                "bucket",
                "file",
                "timestamp_sec",
                "quality",
                "confidence",
            } <= entry.keys()
            jpeg = out_dir / entry["file"]
            assert jpeg.is_file()
            assert jpeg.stat().st_size > 100

    def test_manifest_quality_fields_present(
        self, tmp_path: Path, varied_video: Path, monkeypatch: pytest.MonkeyPatch
    ):
        shots_path, out_dir = self._setup(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            ["findit-keyframe", "extract", str(varied_video), str(shots_path), str(out_dir)],
        )
        assert main() == 0
        manifest = json.loads((out_dir / "manifest.json").read_text())
        q = manifest["keyframes"][0]["quality"]
        assert {
            "laplacian_var",
            "mean_luma",
            "luma_variance",
            "entropy",
            "saliency_mass",
        } <= q.keys()

    def test_jpeg_round_trips(
        self, tmp_path: Path, varied_video: Path, monkeypatch: pytest.MonkeyPatch
    ):
        shots_path, out_dir = self._setup(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            ["findit-keyframe", "extract", str(varied_video), str(shots_path), str(out_dir)],
        )
        assert main() == 0
        manifest = json.loads((out_dir / "manifest.json").read_text())
        # Decode the first JPEG back through PyAV; succeeds iff the file is valid.
        jpeg = out_dir / manifest["keyframes"][0]["file"]
        with av.open(str(jpeg)) as container:
            frame = next(container.decode(video=0))
            assert frame.width > 0
            assert frame.height > 0

    def test_filename_pattern_uses_shot_and_bucket(
        self, tmp_path: Path, varied_video: Path, monkeypatch: pytest.MonkeyPatch
    ):
        shots_path, out_dir = self._setup(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            ["findit-keyframe", "extract", str(varied_video), str(shots_path), str(out_dir)],
        )
        assert main() == 0
        manifest = json.loads((out_dir / "manifest.json").read_text())
        for entry in manifest["keyframes"]:
            expected = f"kf_{entry['shot_id']:03d}_{entry['bucket']:03d}.jpg"
            assert entry["file"] == expected


# --------------------------------------------------------------------------- #
# Exit codes                                                                  #
# --------------------------------------------------------------------------- #


class TestExitCodes:
    def test_bad_shots_json_returns_input_error(
        self, tmp_path: Path, varied_video: Path, monkeypatch: pytest.MonkeyPatch
    ):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        monkeypatch.setattr(
            sys,
            "argv",
            ["findit-keyframe", "extract", str(varied_video), str(bad), str(tmp_path / "out")],
        )
        assert main() == 1

    def test_unknown_config_field_returns_input_error(
        self, tmp_path: Path, varied_video: Path, monkeypatch: pytest.MonkeyPatch
    ):
        shots = tmp_path / "shots.json"
        _write_shots_json(
            shots,
            [{"id": 0, "start_pts": 0, "end_pts": 1000, "timebase_num": 1, "timebase_den": 1000}],
        )
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"unknown_field": 42}))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "findit-keyframe",
                "extract",
                str(varied_video),
                str(shots),
                str(tmp_path / "out"),
                "--config",
                str(cfg),
            ],
        )
        assert main() == 1

    def test_missing_video_returns_extraction_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        shots = tmp_path / "shots.json"
        _write_shots_json(
            shots,
            [{"id": 0, "start_pts": 0, "end_pts": 1000, "timebase_num": 1, "timebase_den": 1000}],
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "findit-keyframe",
                "extract",
                str(tmp_path / "nope.mp4"),
                str(shots),
                str(tmp_path / "out"),
            ],
        )
        assert main() == 2

    def test_saliency_apple_stub_returns_input_error(
        self, tmp_path: Path, varied_video: Path, monkeypatch: pytest.MonkeyPatch
    ):
        shots = tmp_path / "shots.json"
        _write_shots_json(
            shots,
            [{"id": 0, "start_pts": 0, "end_pts": 1000, "timebase_num": 1, "timebase_den": 1000}],
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "findit-keyframe",
                "extract",
                str(varied_video),
                str(shots),
                str(tmp_path / "out"),
                "--saliency",
                "apple",
            ],
        )
        # T6 (Apple Vision saliency) lands in P4; until then the CLI rejects
        # the choice with a clear message rather than silently dropping it.
        assert main() == 1
