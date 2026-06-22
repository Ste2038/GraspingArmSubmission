#!/usr/bin/env python3
"""CLI for reconstructing canonical and extension ICG-Net training datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.icg_dataset_lib import (
    CANONICAL_LAYOUT,
    OFFICIAL_URDF_ROOT,
    canonical_shard_names,
    ensure_dir,
    register_extension_objects,
    resolve_icg_data_root,
    resolve_icg_object_root,
    run_raw_generation,
    write_processed_shard,
    write_reconstruction_audit,
)


DEFAULT_FULL_NUM_GRASPS = 2_000_000
DEFAULT_PILOT_NUM_GRASPS = 240
SCENE_SEED_OFFSETS = {"packed": 0, "pile": 10_000}
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000, "test": 2_000}


def comma_separated_choices(value: str, allowed: set[str], label: str) -> list[str]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        raise argparse.ArgumentTypeError(f"{label} must not be empty.")
    if raw == ["all"]:
        return sorted(item for item in allowed if item != "all")
    invalid = [item for item in raw if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported {label}: {', '.join(invalid)}")
    return raw


def comma_separated_ints(value: str) -> list[int]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        raise argparse.ArgumentTypeError("Shard indices must not be empty.")
    try:
        return [int(item) for item in raw]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid shard index list: {value}") from exc


def select_shard_names(shard_names: list[str], requested_indices: list[int] | None, label: str) -> list[str]:
    if requested_indices is None:
        return shard_names
    invalid = sorted(index for index in requested_indices if index < 0 or index >= len(shard_names))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported {label} shard indices: {', '.join(str(index) for index in invalid)}"
        )
    seen: set[int] = set()
    ordered = []
    for index in requested_indices:
        if index in seen:
            continue
        seen.add(index)
        ordered.append(shard_names[index])
    return ordered


def shard_index_from_name(shard_name: str) -> int:
    try:
        return int(shard_name.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse canonical shard index from: {shard_name}") from exc


def scene_seed_offset(scene: str) -> int:
    if scene not in SCENE_SEED_OFFSETS:
        raise ValueError(f"Unsupported scene for stable seed offset: {scene}")
    return SCENE_SEED_OFFSETS[scene]


def canonical_shard_seed(base_seed: int, scene: str, shard_name: str) -> int:
    return base_seed + scene_seed_offset(scene) + shard_index_from_name(shard_name) * 101


def extension_shard_seed(base_seed: int, scene: str, split: str, shard_index: int) -> int:
    if split not in SPLIT_SEED_OFFSETS:
        raise ValueError(f"Unsupported split for stable seed offset: {split}")
    return base_seed + scene_seed_offset(scene) + SPLIT_SEED_OFFSETS[split] + shard_index * 101


def default_num_grasps(stage: str, requested: int | None) -> int:
    if requested is not None:
        return requested
    return DEFAULT_PILOT_NUM_GRASPS if stage == "pilot" else DEFAULT_FULL_NUM_GRASPS


def default_num_proc(requested: int | None, num_grasps: int, grasps_per_scene: int) -> int:
    if requested is not None:
        return requested
    max_workers = max(1, num_grasps // max(1, grasps_per_scene))
    if max_workers <= 2:
        return 1
    return max(1, min(os.cpu_count() or 1, 32, max_workers))


def write_summary(base_dir: Path, stem: str, payload: dict[str, Any]) -> dict[str, Path]:
    ensure_dir(base_dir)
    json_path = base_dir / f"{stem}.json"
    md_path = base_dir / f"{stem}.md"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"# {payload['title']}",
        "",
        f"- Track: `{payload['track']}`",
    ]
    if "stage" in payload:
        lines.append(f"- Stage: `{payload['stage']}`")
    if "steps" in payload:
        lines.append(f"- Steps: `{', '.join(payload['steps'])}`")
    lines.extend(
        [
            f"- Data root: `{payload['data_root']}`",
            f"- Reconstruction version: `{payload['reconstruction_version']}`",
            "",
            "## Shards",
            "",
            "| Name | Scene | Split/Object Set | Raw Root | Processed Root | Scene Count | Splits | Status |",
            "| --- | --- | --- | --- | --- | ---: | --- | --- |",
        ]
    )

    for row in payload["rows"]:
        lines.append(
            "| {name} | {scene} | {scope} | `{raw_root}` | `{processed_root}` | {scene_count} | {split_counts} | {status} |".format(
                name=row["name"],
                scene=row["scene"],
                scope=row["scope"],
                raw_root=row["raw_root"],
                processed_root=row["processed_root"],
                scene_count=row["scene_count"],
                split_counts=row["split_counts"],
                status=row["status"],
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def add_shared_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=None, help="Override ICG_DATA_ROOT for this command.")
    parser.add_argument("--num-grasps", type=int, default=None)
    parser.add_argument("--grasps-per-scene", type=int, default=120)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=1)
    parser.add_argument("--object-count-lambda", type=int, default=5)
    parser.add_argument("--num-rotations", type=int, default=12)
    parser.add_argument("--horizontal-percentile", type=float, default=0.85)
    parser.add_argument("--urdf-root", type=Path, default=None)
    parser.add_argument("--category-map", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--num-sdf-points", type=int, default=32768)
    parser.add_argument("--num-grasp-points-scene", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--reconstruction-version", type=str, default="best-faith-v1")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-based", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-furthest", action=argparse.BooleanOptionalAction, default=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Write the reconstruction audit files.")
    audit_parser.add_argument("--data-root", type=Path, default=None)
    audit_parser.add_argument("--output-dir", type=Path, default=None)
    audit_parser.add_argument(
        "--reconstruction-version",
        type=str,
        default="best-faith-v1",
        help="Version label included in the audit summary path metadata.",
    )

    raw_parser = subparsers.add_parser("generate-raw", help="Generate one raw shard from the published simulator.")
    raw_parser.add_argument("--raw-root", type=Path, required=True)
    raw_parser.add_argument("--scene", choices=("packed", "pile"), required=True)
    raw_parser.add_argument("--object-set", type=str, required=True)
    add_shared_generation_args(raw_parser)

    process_parser = subparsers.add_parser("process-shard", help="Convert one raw shard into processed samples.")
    process_parser.add_argument("--raw-root", type=Path, required=True)
    process_parser.add_argument("--output-root", type=Path, required=True)
    process_parser.add_argument("--category-map", type=Path, default=None)
    process_parser.add_argument("--voxel-size", type=float, default=0.003)
    process_parser.add_argument("--num-sdf-points", type=int, default=32768)
    process_parser.add_argument("--num-grasp-points-scene", type=int, default=128)
    process_parser.add_argument("--val-fraction", type=float, default=0.1)
    process_parser.add_argument("--reconstruction-version", type=str, default="best-faith-v1")
    process_parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    canonical_parser = subparsers.add_parser("canonical", help="Run canonical pilot/full reconstruction.")
    canonical_parser.add_argument(
        "--stage",
        choices=("pilot", "full"),
        default="pilot",
        help="Pilot generates one reduced shard per scene family before full-scale generation.",
    )
    canonical_parser.add_argument(
        "--scene",
        type=lambda value: comma_separated_choices(value, {"packed", "pile", "all"}, "scene"),
        default=["packed", "pile"],
        help="Scene families to include: packed, pile, or all.",
    )
    canonical_parser.add_argument(
        "--steps",
        type=lambda value: comma_separated_choices(value, {"raw", "process", "all"}, "steps"),
        default=["raw", "process"],
        help="Which stages to execute: raw, process, or all.",
    )
    canonical_parser.add_argument(
        "--shard-indices",
        type=comma_separated_ints,
        default=None,
        help="Optional comma-separated shard indices to process per selected scene, e.g. 1,2,3.",
    )
    add_shared_generation_args(canonical_parser)

    extension_catalog_parser = subparsers.add_parser(
        "register-extension-objects",
        help="Split custom object packages into packed_ext/pile_ext train/val/test catalogs.",
    )
    extension_catalog_parser.add_argument("--source-root", type=Path, required=True)
    extension_catalog_parser.add_argument("--catalog-root", type=Path, default=None)
    extension_catalog_parser.add_argument("--scene-scopes", nargs="+", default=["packed", "pile"])
    extension_catalog_parser.add_argument("--symlink", action=argparse.BooleanOptionalAction, default=True)

    extension_parser = subparsers.add_parser("extension", help="Generate extension-track raw/processed shards.")
    extension_parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    extension_parser.add_argument(
        "--scene",
        type=lambda value: comma_separated_choices(value, {"packed", "pile", "all"}, "scene"),
        default=["packed", "pile"],
    )
    extension_parser.add_argument(
        "--split",
        type=lambda value: comma_separated_choices(value, {"train", "val", "test", "all"}, "split"),
        default=["train", "val", "test"],
    )
    extension_parser.add_argument(
        "--steps",
        type=lambda value: comma_separated_choices(value, {"raw", "process", "all"}, "steps"),
        default=["raw", "process"],
    )
    extension_parser.add_argument(
        "--shards-per-split",
        type=int,
        default=1,
        help="Number of extension shards to generate per scene/split scope.",
    )
    extension_parser.add_argument("--catalog-root", type=Path, default=None)
    add_shared_generation_args(extension_parser)

    return parser.parse_args()


def do_audit(args: argparse.Namespace) -> None:
    data_root = resolve_icg_data_root(args.data_root)
    output_dir = args.output_dir or data_root / "canonical" / "audit"
    outputs = write_reconstruction_audit(output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


def do_generate_raw(args: argparse.Namespace) -> None:
    num_grasps = default_num_grasps("pilot", args.num_grasps)
    manifest = run_raw_generation(
        args.raw_root,
        scene=args.scene,
        object_set=args.object_set,
        urdf_root=args.urdf_root or OFFICIAL_URDF_ROOT,
        num_grasps=num_grasps,
        grasps_per_scene=args.grasps_per_scene,
        num_proc=default_num_proc(args.num_proc, num_grasps, args.grasps_per_scene),
        seed=args.seed,
        num_views=args.num_views,
        contact_based=args.contact_based,
        sample_furthest=args.sample_furthest,
        num_rotations=args.num_rotations,
        horizontal_percentile=args.horizontal_percentile,
        object_count_lambda=args.object_count_lambda,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def do_process_shard(args: argparse.Namespace) -> None:
    manifest = write_processed_shard(
        args.raw_root,
        args.output_root,
        voxel_size=args.voxel_size,
        num_sdf_points=args.num_sdf_points,
        num_grasp_points_scene=args.num_grasp_points_scene,
        val_fraction=args.val_fraction,
        category_map_path=args.category_map,
        reconstruction_version=args.reconstruction_version,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def do_canonical(args: argparse.Namespace) -> None:
    data_root = resolve_icg_data_root(args.data_root)
    scenes = args.scene
    steps = args.steps
    num_grasps = default_num_grasps(args.stage, args.num_grasps)
    num_proc = default_num_proc(args.num_proc, num_grasps, args.grasps_per_scene)
    urdf_root = args.urdf_root or OFFICIAL_URDF_ROOT

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        object_set = CANONICAL_LAYOUT[scene]["object_set"]
        selected_shards = select_shard_names(
            canonical_shard_names(scene, stage=args.stage),
            args.shard_indices,
            label=scene,
        )
        for shard_name in selected_shards:
            raw_root = data_root / "canonical" / "raw" / shard_name
            processed_root = data_root / "canonical" / "processed" / shard_name
            scope = object_set
            status = "pending"
            if "raw" in steps:
                run_raw_generation(
                    raw_root,
                    scene=scene,
                    object_set=object_set,
                    urdf_root=urdf_root,
                    num_grasps=num_grasps,
                    grasps_per_scene=args.grasps_per_scene,
                    num_proc=num_proc,
                    seed=canonical_shard_seed(args.seed, scene, shard_name),
                    num_views=args.num_views,
                    contact_based=args.contact_based,
                    sample_furthest=args.sample_furthest,
                    num_rotations=args.num_rotations,
                    horizontal_percentile=args.horizontal_percentile,
                    object_count_lambda=args.object_count_lambda,
                    resume=args.resume,
                )
                status = "raw"
            scene_count = 0
            split_counts = "-"
            if "process" in steps:
                manifest = write_processed_shard(
                    raw_root,
                    processed_root,
                    voxel_size=args.voxel_size,
                    num_sdf_points=args.num_sdf_points,
                    num_grasp_points_scene=args.num_grasp_points_scene,
                    val_fraction=args.val_fraction,
                    category_map_path=args.category_map,
                    reconstruction_version=args.reconstruction_version,
                    resume=args.resume,
                )
                scene_count = int(manifest["scene_count"])
                split_counts = json.dumps(manifest["split_counts"], sort_keys=True)
                status = manifest["reconstruction_status"]
            rows.append(
                {
                    "name": shard_name,
                    "scene": scene,
                    "scope": scope,
                    "raw_root": str(raw_root),
                    "processed_root": str(processed_root),
                    "scene_count": scene_count,
                    "split_counts": split_counts,
                    "status": status,
                }
            )

    summary = {
        "title": "ICG-Net Canonical Reconstruction Summary",
        "track": "canonical",
        "stage": args.stage,
        "steps": steps,
        "data_root": str(data_root),
        "reconstruction_version": args.reconstruction_version,
        "rows": rows,
    }
    outputs = write_summary(data_root / "canonical" / "reports", f"canonical_{args.stage}", summary)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


def do_register_extension_objects(args: argparse.Namespace) -> None:
    manifest = register_extension_objects(
        args.source_root,
        catalog_root=args.catalog_root,
        scene_scopes=tuple(args.scene_scopes),
        symlink=args.symlink,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def do_extension(args: argparse.Namespace) -> None:
    data_root = resolve_icg_data_root(args.data_root)
    object_root = resolve_icg_object_root(args.catalog_root)
    scenes = args.scene
    splits = args.split
    steps = args.steps
    num_grasps = default_num_grasps(args.stage, args.num_grasps)
    num_proc = default_num_proc(args.num_proc, num_grasps, args.grasps_per_scene)

    rows: list[dict[str, Any]] = []
    for scene in scenes:
        for split in splits:
            object_set = f"{scene}_ext/{split}"
            for shard_idx in range(args.shards_per_split):
                shard_name = f"{scene}_ext_{split}_contact_{shard_idx}"
                raw_root = data_root / "extension" / "raw" / shard_name
                processed_root = data_root / "extension" / "processed" / shard_name
                status = "pending"
                if "raw" in steps:
                    run_raw_generation(
                        raw_root,
                        scene=scene,
                        object_set=object_set,
                        urdf_root=object_root,
                        num_grasps=num_grasps,
                        grasps_per_scene=args.grasps_per_scene,
                        num_proc=num_proc,
                        seed=extension_shard_seed(args.seed, scene, split, shard_idx),
                        num_views=args.num_views,
                        contact_based=args.contact_based,
                        sample_furthest=args.sample_furthest,
                        num_rotations=args.num_rotations,
                        horizontal_percentile=args.horizontal_percentile,
                        object_count_lambda=args.object_count_lambda,
                        resume=args.resume,
                    )
                    status = "raw"
                scene_count = 0
                split_counts = "-"
                if "process" in steps:
                    manifest = write_processed_shard(
                        raw_root,
                        processed_root,
                        voxel_size=args.voxel_size,
                        num_sdf_points=args.num_sdf_points,
                        num_grasp_points_scene=args.num_grasp_points_scene,
                        val_fraction=args.val_fraction,
                        category_map_path=args.category_map,
                        reconstruction_version=args.reconstruction_version,
                        resume=args.resume,
                    )
                    scene_count = int(manifest["scene_count"])
                    split_counts = json.dumps(manifest["split_counts"], sort_keys=True)
                    status = manifest["reconstruction_status"]
                rows.append(
                    {
                        "name": shard_name,
                        "scene": scene,
                        "scope": object_set,
                        "raw_root": str(raw_root),
                        "processed_root": str(processed_root),
                        "scene_count": scene_count,
                        "split_counts": split_counts,
                        "status": status,
                    }
                )

    summary = {
        "title": "ICG-Net Extension Reconstruction Summary",
        "track": "extension",
        "stage": args.stage,
        "steps": steps,
        "data_root": str(data_root),
        "reconstruction_version": args.reconstruction_version,
        "rows": rows,
    }
    outputs = write_summary(data_root / "extension" / "reports", f"extension_{args.stage}", summary)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        do_audit(args)
    elif args.command == "generate-raw":
        do_generate_raw(args)
    elif args.command == "process-shard":
        do_process_shard(args)
    elif args.command == "canonical":
        do_canonical(args)
    elif args.command == "register-extension-objects":
        do_register_extension_objects(args)
    elif args.command == "extension":
        do_extension(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
