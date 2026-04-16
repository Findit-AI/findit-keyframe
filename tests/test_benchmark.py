"""Smoke test for ``benchmarks/bench_e2e.py``.

The script is meant to be run as a CLI; we exercise it via ``subprocess`` to
verify it boots, writes a row to ``results.md``, and exits 0 on a tiny
fixture video. Numerical thresholds belong in the script's output, not in
this test — we only check structure.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


_BENCH_SCRIPT = "benchmarks/bench_e2e.py"


def test_bench_runs_and_writes_results_md(tmp_path: Path, varied_video: Path):
    results_md = tmp_path / "results.md"
    proc = subprocess.run(
        [
            sys.executable,
            _BENCH_SCRIPT,
            "--video",
            str(varied_video),
            "--results-md",
            str(results_md),
            "--target-size",
            "64",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"

    summary = json.loads(proc.stdout)
    assert summary["video"] == str(varied_video)
    assert summary["wall_sec"] > 0
    assert summary["keyframes"] >= 1
    assert summary["target_size"] == 64

    assert results_md.is_file()
    content = results_md.read_text()
    assert "findit-keyframe benchmarks" in content
    assert varied_video.name in content


def test_bench_quiet_suppresses_stdout(tmp_path: Path, varied_video: Path):
    results_md = tmp_path / "results.md"
    proc = subprocess.run(
        [
            sys.executable,
            _BENCH_SCRIPT,
            "--video",
            str(varied_video),
            "--results-md",
            str(results_md),
            "--target-size",
            "64",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert proc.stdout.strip() == ""
    assert results_md.is_file()  # row still appended
