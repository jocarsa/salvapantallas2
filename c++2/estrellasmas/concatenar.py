#!/usr/bin/env python3
"""
concatenar_fastcopy.py

Fixes concat list quoting for paths with spaces.
Randomly concatenates MP4 clips until ~10 hours using ffmpeg concat demuxer + -c copy (FAST).
"""

import argparse
import random
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import json


def run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out.decode("utf-8", "replace"))
    return float(data["format"]["duration"])


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def concat_escape(path: str) -> str:
    """
    concat demuxer syntax: file '...'
    If path contains single quotes, escape them: ' -> '\'' (close, escape, reopen)
    """
    return path.replace("'", r"'\''")


def pick_playlist(files: List[Path], durs: List[float], target_seconds: float) -> List[Tuple[Path, float]]:
    playlist: List[Tuple[Path, float]] = []
    total = 0.0
    n = len(files)
    while total < target_seconds:
        i = random.randrange(n)
        playlist.append((files[i], durs[i]))
        total += durs[i]
    return playlist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mkv", action="store_true", help="Output MKV (more tolerant than MP4 for copy-concat).")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cwd = Path(".").resolve()
    files = sorted([p for p in cwd.glob(args.glob) if p.is_file()])
    if not files:
        print(f"No files found with glob: {args.glob}")
        return 1

    print(f"Found {len(files)} files. Probing durations...")
    good_files: List[Path] = []
    durs: List[float] = []
    for p in files:
        try:
            d = ffprobe_duration(p)
            if d > 0:
                good_files.append(p.resolve())
                durs.append(d)
        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    if not good_files:
        print("No usable files after probing.")
        return 1

    target_seconds = args.hours * 3600.0
    playlist = pick_playlist(good_files, durs, target_seconds)
    total = sum(d for _, d in playlist)
    print(f"Playlist clips: {len(playlist)} | Target: {target_seconds:.1f}s | Actual: {total:.1f}s ({total/3600.0:.3f}h)")

    stamp = now_stamp()
    ext = "mkv" if args.mkv else "mp4"
    out_name = args.out or f"random_{int(args.hours)}h_{stamp}.{ext}"
    out_path = (cwd / out_name).resolve()

    list_path = cwd / f".concat_{stamp}.txt"
    with list_path.open("w", encoding="utf-8", newline="\n") as f:
        for p, _d in playlist:
            # concat demuxer: file 'ABSOLUTE_PATH'
            ps = concat_escape(str(p))
            f.write(f"file '{ps}'\n")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-v", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
        ]
        if not args.mkv:
            cmd += ["-movflags", "+faststart", "-tag:v", "hvc1"]
        cmd += [str(out_path)]

        run(cmd)
        print(f"\nDONE: {out_path}")
        return 0
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except TypeError:
            if list_path.exists():
                list_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
