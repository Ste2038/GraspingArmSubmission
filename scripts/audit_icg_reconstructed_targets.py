#!/usr/bin/env python3
"""Audit reconstructed ICG-Net grasp targets for best-faith training diagnostics."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.icg_dataset_lib import ensure_dir, resolve_icg_data_root


ORIENTATION_COUNT = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Processed shard root. Defaults to <data-root>/canonical/processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory. Defaults to <data-root>/canonical/reports/target_audit.",
    )
    parser.add_argument(
        "--shard",
        action="append",
        default=[],
        help="Optional shard name to include. Repeat for multiple shards.",
    )
    parser.add_argument("--zero-width-eps", type=float, default=1e-8)
    return parser.parse_args()


def torch_load_cpu(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def scene_family_from_shard(shard_name: str) -> str:
    if shard_name.startswith("packed"):
        return "packed"
    if shard_name.startswith("pile"):
        return "pile"
    return "unknown"


def iter_manifest_rows(processed_root: Path, shard_filter: set[str] | None = None) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    shard_roots = (
        [processed_root]
        if (processed_root / "manifest.json").exists()
        else sorted(path for path in processed_root.iterdir() if path.is_dir())
    )
    for shard_root in shard_roots:
        if shard_filter and shard_root.name not in shard_filter:
            continue
        manifest_path = shard_root / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest.get("scenes", []):
            rows.append((shard_root, row))
    return rows


def class_count_string(labels: torch.Tensor) -> str:
    if labels.numel() == 0:
        return ""
    values, counts = labels.long().unique(return_counts=True)
    pairs = sorted(zip(values.tolist(), counts.tolist()), key=lambda item: item[0])
    return ";".join(f"{int(value)}:{int(count)}" for value, count in pairs)


def orientation_count_string(counts: torch.Tensor) -> str:
    values = [int(value) for value in counts.tolist()]
    return ";".join(f"{idx}:{value}" for idx, value in enumerate(values))


def finite_or_none(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number != number:
        return None
    return number


def audit_sample(shard_root: Path, manifest_row: dict[str, Any], zero_width_eps: float) -> dict[str, Any]:
    sample_path = shard_root / manifest_row["sample_path"]
    sample = torch_load_cpu(sample_path)
    targets = sample["targets"]
    labels = targets["scene_centric_labels"].float()
    class_labels = targets["labels"].long()
    scene_id = str(manifest_row["scene_id"])
    split = str(manifest_row.get("split", "unknown"))

    if labels.ndim != 3 or labels.shape[-1] < ORIENTATION_COUNT + 1:
        raise ValueError(f"Unexpected scene_centric_labels shape in {sample_path}: {tuple(labels.shape)}")

    orientations = labels[..., :ORIENTATION_COUNT]
    widths = labels[..., ORIENTATION_COUNT]
    positive_orientations = orientations > 0
    positive_instance_points = positive_orientations.any(dim=-1)
    positive_scene_points = positive_orientations.any(dim=(0, 2))
    positive_widths = widths[positive_instance_points]

    total_instances = int(labels.shape[0])
    total_grasp_points = int(labels.shape[1])
    total_instance_points = int(total_instances * total_grasp_points)
    total_orientation_slots = int(total_instance_points * ORIENTATION_COUNT)
    positive_instance_point_count = int(positive_instance_points.sum().item())
    positive_scene_point_count = int(positive_scene_points.sum().item())
    positive_orientation_count = int(positive_orientations.sum().item())
    zero_width_positive_count = int((positive_widths <= zero_width_eps).sum().item()) if positive_widths.numel() else 0
    nonzero_widths = positive_widths[positive_widths > zero_width_eps]

    width_min = finite_or_none(nonzero_widths.min().item()) if nonzero_widths.numel() else None
    width_mean = finite_or_none(positive_widths.mean().item()) if positive_widths.numel() else None
    width_nonzero_mean = finite_or_none(nonzero_widths.mean().item()) if nonzero_widths.numel() else None
    width_max = finite_or_none(positive_widths.max().item()) if positive_widths.numel() else None
    orientation_counts = positive_orientations.sum(dim=(0, 1))

    return {
        "shard": shard_root.name,
        "scene_family": scene_family_from_shard(shard_root.name),
        "split": split,
        "scene_id": scene_id,
        "sample_path": str(sample_path),
        "num_instances": total_instances,
        "num_points": int(sample.get("raw_coordinates", torch.empty(0)).shape[0]),
        "num_grasp_points": total_grasp_points,
        "total_instance_points": total_instance_points,
        "total_orientation_slots": total_orientation_slots,
        "positive_scene_points": positive_scene_point_count,
        "positive_scene_point_fraction": positive_scene_point_count / total_grasp_points if total_grasp_points else 0.0,
        "positive_instance_points": positive_instance_point_count,
        "positive_instance_point_fraction": (
            positive_instance_point_count / total_instance_points if total_instance_points else 0.0
        ),
        "positive_orientations": positive_orientation_count,
        "positive_orientation_fraction": (
            positive_orientation_count / total_orientation_slots if total_orientation_slots else 0.0
        ),
        "zero_width_positive_points": zero_width_positive_count,
        "zero_width_positive_fraction": (
            zero_width_positive_count / positive_instance_point_count if positive_instance_point_count else 0.0
        ),
        "width_min_nonzero": width_min,
        "width_mean_all_positive": width_mean,
        "width_mean_nonzero": width_nonzero_mean,
        "width_max": width_max,
        "orientation_positive_counts": orientation_count_string(orientation_counts),
        "class_counts": class_count_string(class_labels),
        "status": str(manifest_row.get("status", sample.get("meta", {}).get("status", "unknown"))),
    }


def aggregate_records(records: list[dict[str, Any]], group_name: str, key: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[key(record)].append(record)

    rows: list[dict[str, Any]] = []
    for group_value, group_records in sorted(grouped.items(), key=lambda item: item[0]):
        sample_count = len(group_records)
        total_grasp_points = sum(int(row["num_grasp_points"]) for row in group_records)
        total_instance_points = sum(int(row["total_instance_points"]) for row in group_records)
        total_orientation_slots = sum(int(row["total_orientation_slots"]) for row in group_records)
        positive_scene_points = sum(int(row["positive_scene_points"]) for row in group_records)
        positive_instance_points = sum(int(row["positive_instance_points"]) for row in group_records)
        positive_orientations = sum(int(row["positive_orientations"]) for row in group_records)
        zero_width_positive_points = sum(int(row["zero_width_positive_points"]) for row in group_records)

        width_count = sum(int(row["positive_instance_points"]) for row in group_records)
        width_sum = sum(
            float(row["width_mean_all_positive"]) * int(row["positive_instance_points"])
            for row in group_records
            if row["width_mean_all_positive"] is not None
        )
        nonzero_width_values = [
            float(value)
            for row in group_records
            for value in (row["width_min_nonzero"], row["width_max"])
            if value is not None
        ]

        rows.append(
            {
                "group": group_name,
                "value": group_value,
                "samples": sample_count,
                "positive_scene_point_fraction": (
                    positive_scene_points / total_grasp_points if total_grasp_points else 0.0
                ),
                "positive_instance_point_fraction": (
                    positive_instance_points / total_instance_points if total_instance_points else 0.0
                ),
                "positive_orientation_fraction": (
                    positive_orientations / total_orientation_slots if total_orientation_slots else 0.0
                ),
                "zero_width_positive_points": zero_width_positive_points,
                "zero_width_positive_fraction": (
                    zero_width_positive_points / positive_instance_points if positive_instance_points else 0.0
                ),
                "width_mean_all_positive": width_sum / width_count if width_count else None,
                "width_min_nonzero": min(nonzero_width_values) if nonzero_width_values else None,
                "width_max": max(nonzero_width_values) if nonzero_width_values else None,
            }
        )
    return rows


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": aggregate_records(records, "all", lambda _: "all"),
        "by_scene_family": aggregate_records(records, "scene_family", lambda row: str(row["scene_family"])),
        "by_split": aggregate_records(records, "split", lambda row: str(row["split"])),
        "by_shard": aggregate_records(records, "shard", lambda row: str(row["shard"])),
    }


def fmt(value: Any, precision: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Group | Value | Samples | Positive Scene Points | Positive Orientations | Zero-Width Positives | Width Mean | Width Range |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        width_range = f"{fmt(row['width_min_nonzero'])} .. {fmt(row['width_max'])}"
        lines.append(
            "| {group} | {value} | {samples} | {scene_frac} | {orient_frac} | {zero_frac} ({zero_count}) | {width_mean} | {width_range} |".format(
                group=row["group"],
                value=row["value"],
                samples=row["samples"],
                scene_frac=fmt(row["positive_scene_point_fraction"]),
                orient_frac=fmt(row["positive_orientation_fraction"]),
                zero_frac=fmt(row["zero_width_positive_fraction"]),
                zero_count=row["zero_width_positive_points"],
                width_mean=fmt(row["width_mean_all_positive"]),
                width_range=width_range,
            )
        )
    return lines


def write_outputs(records: list[dict[str, Any]], output_dir: Path, metadata: dict[str, Any]) -> dict[str, Path]:
    output_path = ensure_dir(output_dir)
    summary = build_summary(records)
    payload = {"metadata": metadata, "summary": summary, "records": records}

    json_path = output_path / "target_audit.json"
    csv_path = output_path / "target_audit_samples.csv"
    md_path = output_path / "target_audit.md"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = sorted({key for record in records for key in record.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# ICG-Net Reconstructed Target Audit",
        "",
        f"- Generated: `{metadata['generated_at']}`",
        f"- Processed root: `{metadata['processed_root']}`",
        f"- Shards: `{metadata['shards']}`",
        f"- Samples: `{len(records)}`",
        f"- Reconstruction status: `best-faith`, not `paper-exact`",
        f"- Missing-info decision point: `missing-info-decision-point` / `e75e072`",
        "",
        "This report audits reconstructed grasp-supervision targets so that the",
        "best-faith training track can continue while preserving the assumptions",
        "that must be revisited if author-only information becomes available.",
        "",
        "## Summary",
        "",
    ]
    for section_name in ("all", "by_scene_family", "by_split", "by_shard"):
        lines.append(f"### {section_name.replace('_', ' ').title()}")
        lines.append("")
        lines.extend(markdown_table(summary[section_name]))
        lines.append("")

    zero_width_rows = [row for row in records if int(row["zero_width_positive_points"]) > 0]
    lines.extend(
        [
            "## Flags",
            "",
            f"- Samples with zero-width positive grasp points: `{len(zero_width_rows)}`",
            "- Zero-width positives are not automatically treated as an error because",
            "  the original processed-dataset convention is currently missing.",
            "- If future author information confirms positive grasps must have nonzero",
            "  width, use this report to identify affected samples and rerun training.",
            "",
            "## Per-Sample CSV",
            "",
            f"See `{csv_path.name}` for sample-level orientation counts, class counts,",
            "positive fractions, and width statistics.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def run_audit(
    processed_root: Path,
    output_dir: Path,
    shard_filter: set[str] | None = None,
    zero_width_eps: float = 1e-8,
) -> dict[str, Path]:
    rows = iter_manifest_rows(processed_root, shard_filter)
    if not rows:
        raise FileNotFoundError(f"No processed samples found under {processed_root}")
    records = [audit_sample(shard_root, row, zero_width_eps=zero_width_eps) for shard_root, row in rows]
    metadata = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "processed_root": str(processed_root),
        "shards": ",".join(sorted({shard_root.name for shard_root, _ in rows})),
        "zero_width_eps": zero_width_eps,
    }
    return write_outputs(records, output_dir, metadata)


def main() -> None:
    args = parse_args()
    data_root = resolve_icg_data_root(args.data_root)
    processed_root = (args.processed_root or data_root / "canonical" / "processed").resolve()
    output_dir = (args.output_dir or data_root / "canonical" / "reports" / "target_audit").resolve()
    shard_filter = set(args.shard) if args.shard else None
    outputs = run_audit(
        processed_root=processed_root,
        output_dir=output_dir,
        shard_filter=shard_filter,
        zero_width_eps=args.zero_width_eps,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
