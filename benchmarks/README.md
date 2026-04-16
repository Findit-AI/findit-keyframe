# Benchmarks

End-to-end performance baselines for `findit-keyframe`. The Rust port (P5+)
is expected to beat these numbers by 5-10x on identical inputs.

## Running

```bash
# Synthetic uniform-shot baseline on any video.
python benchmarks/bench_e2e.py --video path/to/video.mp4

# Replay a real shot list (e.g. produced by scenesdetect).
python benchmarks/bench_e2e.py --video path/to/video.mp4 --shots shots.json

# Smaller output frames (faster, cheaper).
python benchmarks/bench_e2e.py --video path/to/video.mp4 --target-size 256
```

Each run appends a row to [`results.md`](results.md) with the date, git
SHA, and headline numbers (wall time, throughput, peak RSS).

## Reading the numbers

* **Wall (s)**: total time for `extract_all`, decoder open + close included.
* **KF/s**: keyframes emitted per wall-clock second. Useful to compare
  across videos of different length.
* **Mem (MB)**: peak resident-set size. Linux reports KB internally;
  macOS reports bytes; the script normalises both to MB.

## Performance budget (P3 baseline target)

Per `TASKS.md` §3.T8: the Kino Demo render (1m44s, 1080p) should finish
extraction in under 30 seconds on an M-series Mac at the default
`target_size=384`. A regression beyond that warrants investigation
before tagging a release.
