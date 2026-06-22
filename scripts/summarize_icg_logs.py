#!/usr/bin/env python3
"""Summarize ICG benchmark JSON logs as a compact Markdown table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def as_percent(value: float | int | None) -> str:
    if value is None:
        return "-"
    number = float(value)
    if abs(number) <= 1.0:
        number *= 100.0
    return f"{number:.2f}"


def scene_from_name(path: Path) -> str:
    match = re.search(r"_(packed|pile|obj|egad)_", path.name)
    return match.group(1) if match else "-"


def load_summary(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "file": path.name,
        "scene": scene_from_name(path),
        "gsr": as_percent(data.get("gsr")),
        "gsr_std": as_percent(data.get("gsr_std")),
        "dr": as_percent(data.get("dr")),
        "dr_std": as_percent(data.get("dr_std")),
        "collisions": data.get("collisions", "-"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logdir", type=Path, nargs="?", default=Path("logs"))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    files = sorted(args.logdir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON logs found in {args.logdir}")

    rows = [load_summary(path) for path in files]

    print("| File | Scene | GSR (%) | GSR Std (%) | DR (%) | DR Std (%) | Collisions |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['file']} | {row['scene']} | {row['gsr']} | {row['gsr_std']} | "
            f"{row['dr']} | {row['dr_std']} | {row['collisions']} |"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8") as handle:
            handle.write("file,scene,gsr_percent,gsr_std_percent,dr_percent,dr_std_percent,collisions\n")
            for row in rows:
                handle.write(
                    f"{row['file']},{row['scene']},{row['gsr']},{row['gsr_std']},"
                    f"{row['dr']},{row['dr_std']},{row['collisions']}\n"
                )


if __name__ == "__main__":
    main()
