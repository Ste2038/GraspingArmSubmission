#!/usr/bin/env python3
"""Plot ICG-Net training metrics and generate a paper-comparison report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.icg_training_lib import generate_training_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_source",
        type=Path,
        help="Path to metrics.jsonl / metrics.csv or the run directory containing them.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--benchmark-logdir", type=Path, default=Path("logs/icg_full"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_training_report(args.metrics_source, args.output_dir, args.benchmark_logdir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
