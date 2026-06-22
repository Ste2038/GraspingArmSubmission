"""ICG-Net dataset reconstruction helpers and local dataset replacements."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET

import numpy as np
import torch
import trimesh
import yaml
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_COLLECTION_SRC = REPO_ROOT / "third_party" / "icg_net" / "data_collection" / "src"
OFFICIAL_CONFIG = REPO_ROOT / "third_party" / "icg_benchmark" / "data" / "icgnet" / "51--0.656" / "config.yaml"
OFFICIAL_URDF_ROOT = REPO_ROOT / "third_party" / "icg_benchmark" / "data" / "urdfs"
RAW_GENERATOR = REPO_ROOT / "third_party" / "icg_net" / "data_collection" / "scripts" / "generate_data_parallel.py"

if str(DATA_COLLECTION_SRC) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_SRC))


DEFAULT_SAMPLE_SCHEMA = {
    "target_keys": ["labels", "masks", "sdf", "scene_centric_labels", "object_centric_labels"],
    "mask_type": "masks",
    "grasp_label_layout": "12 orientation logits + 1 width",
    "sdf_target_layout": "per-instance occupancy sign encoded as +/-1 over shared query points",
}

CANONICAL_LAYOUT = {
    "packed": {"shards": 5, "object_set": "packed/train"},
    "pile": {"shards": 10, "object_set": "pile/train"},
}


def resolve_icg_data_root(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    env_path = os.environ.get("ICG_DATA_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (REPO_ROOT / "data" / "icg_reconstructed").resolve()


def resolve_icg_object_root(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    env_path = os.environ.get("ICG_OBJECT_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (resolve_icg_data_root() / "objects").resolve()


def canonical_shard_names(scene: str, stage: str = "pilot") -> list[str]:
    shard_count = 1 if stage == "pilot" else CANONICAL_LAYOUT[scene]["shards"]
    return [f"{scene}_2M_contact_{index}" for index in range(shard_count)]


def canonical_processed_dirs(stage: str = "pilot", data_root: str | Path | None = None) -> list[Path]:
    root = resolve_icg_data_root(data_root) / "canonical" / "processed"
    return [root / name for scene in ("packed", "pile") for name in canonical_shard_names(scene, stage=stage)]


def load_optional_category_map(path: str | Path | None) -> dict[str, int]:
    if path is None:
        return {}
    category_path = Path(path)
    if not category_path.exists():
        raise FileNotFoundError(f"Category map not found: {category_path}")
    with category_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Category map must be a JSON object of object-name -> class-id.")
    return {str(key): int(value) for key, value in data.items()}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): sha256_path(path)
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    }


def hash_scene_id(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def assign_scene_splits(scene_ids: Sequence[str], val_fraction: float = 0.1) -> dict[str, str]:
    if not scene_ids:
        return {}
    ordered = sorted(scene_ids, key=lambda scene_id: (hash_scene_id(scene_id), scene_id))
    val_count = max(1, round(len(ordered) * val_fraction)) if len(ordered) > 1 else 0
    val_ids = set(ordered[:val_count])
    return {scene_id: ("val" if scene_id in val_ids else "train") for scene_id in ordered}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_json_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(inner) for inner in value]
    return value


def load_raw_generator_info(raw_root: Path) -> dict[str, Any]:
    info_path = raw_root / "info.yaml"
    if not info_path.exists():
        return {}
    try:
        with info_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.constructor.ConstructorError:
        with info_path.open("r", encoding="utf-8") as handle:
            data = yaml.load(handle, Loader=yaml.UnsafeLoader) or {}
    return normalize_json_value(data) if isinstance(data, dict) else {}


def list_raw_scene_ids(raw_root: Path) -> list[str]:
    pointcloud_dir = raw_root / "pointcloud"
    mesh_pose_dir = raw_root / "mesh_pose_list"
    scene_dir = raw_root / "scenes"
    if not pointcloud_dir.exists() or not mesh_pose_dir.exists():
        return []

    pointcloud_ids = {path.stem for path in pointcloud_dir.glob("*.npz")}
    mesh_ids = {path.stem for path in mesh_pose_dir.glob("*.npz")}
    scene_ids = {path.stem for path in scene_dir.glob("*.npz")} if scene_dir.exists() else pointcloud_ids
    return sorted(pointcloud_ids & mesh_ids & scene_ids)


def load_candidate_rows(raw_root: Path) -> dict[str, list[dict[str, str]]]:
    rows_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    candidate_path = raw_root / "grasps_candidate.csv"
    if not candidate_path.exists():
        return rows_by_scene
    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scene_id = str(row.get("scene_id", "")).strip()
            if scene_id:
                rows_by_scene[scene_id].append(row)
    return rows_by_scene


def load_mesh_pose_list(path: Path) -> list[tuple[Any, Any, Any]]:
    return list(np.load(path, allow_pickle=True)["pc"])


def majority_vote(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    counter = Counter(int(value) for value in values.tolist())
    return counter.most_common(1)[0][0]


def quantize_points(
    coords: np.ndarray,
    colors: np.ndarray | None,
    normals: np.ndarray | None,
    instance_ids: np.ndarray,
    voxel_size: float,
) -> dict[str, np.ndarray]:
    quantized = np.floor(coords / voxel_size + 1e-6).astype(np.int32)
    unique_coords, inverse = np.unique(quantized, axis=0, return_inverse=True)
    counts = np.bincount(inverse)

    raw_coords = np.zeros((len(unique_coords), 3), dtype=np.float32)
    np.add.at(raw_coords, inverse, coords.astype(np.float32))
    raw_coords /= counts[:, None]

    voxel_colors = None
    if colors is not None:
        voxel_colors = np.zeros((len(unique_coords), colors.shape[1]), dtype=np.float32)
        np.add.at(voxel_colors, inverse, colors.astype(np.float32))
        voxel_colors /= counts[:, None]

    voxel_normals = None
    if normals is not None:
        voxel_normals = np.zeros((len(unique_coords), normals.shape[1]), dtype=np.float32)
        np.add.at(voxel_normals, inverse, normals.astype(np.float32))
        norms = np.linalg.norm(voxel_normals, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        voxel_normals = voxel_normals / norms

    voxel_instances = np.zeros(len(unique_coords), dtype=np.int64)
    for voxel_idx in range(len(unique_coords)):
        voxel_instances[voxel_idx] = majority_vote(instance_ids[inverse == voxel_idx])

    return {
        "quantized_coords": unique_coords.astype(np.int32),
        "raw_coordinates": raw_coords.astype(np.float32),
        "colors": voxel_colors.astype(np.float32) if voxel_colors is not None else np.zeros((len(unique_coords), 3), dtype=np.float32),
        "normals": voxel_normals.astype(np.float32) if voxel_normals is not None else np.zeros((len(unique_coords), 3), dtype=np.float32),
        "instance_ids": voxel_instances.astype(np.int64),
    }


def map_pointcloud_instances(pointcloud: np.ndarray, full_pointcloud: np.ndarray, full_instances: np.ndarray) -> np.ndarray:
    if len(full_pointcloud) == 0:
        return np.zeros(len(pointcloud), dtype=np.int64)
    tree = cKDTree(full_pointcloud.astype(np.float64))
    _, indices = tree.query(pointcloud.astype(np.float64), k=1)
    return full_instances[indices].astype(np.int64)


def choose_scene_points(
    candidate_rows: Sequence[dict[str, str]],
    fallback_points: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, str] | None]]:
    rng = np.random.default_rng(seed)
    positive = [row for row in candidate_rows if any(float(row.get(f"label_{idx}", 0.0)) > 0.0 for idx in range(12))]
    negative = [row for row in candidate_rows if row not in positive]

    selected: list[dict[str, str]] = []
    half = count // 2
    if positive:
        rng.shuffle(positive)
        selected.extend(positive[: min(len(positive), half if negative else count)])
    if negative and len(selected) < count:
        rng.shuffle(negative)
        selected.extend(negative[: count - len(selected)])
    if positive and len(selected) < count:
        remaining_positive = [row for row in positive if row not in selected]
        selected.extend(remaining_positive[: count - len(selected)])

    scene_points = np.zeros((count, 3), dtype=np.float32)
    row_refs: list[dict[str, str] | None] = [None] * count
    for idx, row in enumerate(selected[:count]):
        scene_points[idx] = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float32)
        row_refs[idx] = row

    if len(selected) < count:
        if len(fallback_points) == 0:
            fallback_points = np.zeros((1, 3), dtype=np.float32)
        sample_indices = rng.choice(
            np.arange(len(fallback_points)),
            size=count - len(selected),
            replace=len(fallback_points) < (count - len(selected)),
        )
        scene_points[len(selected) :] = fallback_points[sample_indices].astype(np.float32)

    return scene_points, row_refs


def infer_target_instance_from_point(point: np.ndarray, raw_coordinates: np.ndarray, voxel_instance_ids: np.ndarray) -> int | None:
    if len(raw_coordinates) == 0:
        return None
    tree = cKDTree(raw_coordinates.astype(np.float64))
    _, index = tree.query(point.astype(np.float64), k=1)
    instance_id = int(voxel_instance_ids[index])
    return instance_id if instance_id > 0 else None


def build_scene_grasp_targets(
    candidate_rows: Sequence[dict[str, str]],
    raw_coordinates: np.ndarray,
    voxel_instance_ids: np.ndarray,
    instance_id_map: dict[int, int],
    grasp_point_count: int,
    scene_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene_points, row_refs = choose_scene_points(candidate_rows, raw_coordinates, grasp_point_count, seed=hash_scene_id(scene_id))
    labels = np.zeros((len(instance_id_map), grasp_point_count, 13), dtype=np.float32)
    object_labels = np.zeros_like(labels)

    for point_idx, row in enumerate(row_refs):
        if row is None:
            continue
        target_instance = int(float(row.get("target", 0) or 0))
        if target_instance not in instance_id_map:
            inferred = infer_target_instance_from_point(scene_points[point_idx], raw_coordinates, voxel_instance_ids)
            if inferred is None:
                continue
            target_instance = inferred
        local_instance = instance_id_map[target_instance]
        orientation_labels = np.array([float(row.get(f"label_{idx}", 0.0) or 0.0) for idx in range(12)], dtype=np.float32)
        labels[local_instance, point_idx, :12] = orientation_labels
        object_labels[local_instance, point_idx, :12] = orientation_labels
        width = float(row.get("width", 0.0) or 0.0)
        if orientation_labels.sum() > 0:
            labels[local_instance, point_idx, 12] = width
            object_labels[local_instance, point_idx, 12] = width

    return scene_points.astype(np.float32), labels.astype(np.float32), object_labels.astype(np.float32)


def build_instance_masks(voxel_instance_ids: np.ndarray, unique_instances: Sequence[int]) -> np.ndarray:
    if not unique_instances:
        return np.zeros((0, len(voxel_instance_ids)), dtype=np.float32)
    return np.stack([(voxel_instance_ids == instance_id).astype(np.float32) for instance_id in unique_instances], axis=0)


def build_semantic_labels(unique_instances: Sequence[int], category_map: dict[str, int]) -> np.ndarray:
    default_label = int(category_map.get("default", 0))
    return np.full((len(unique_instances),), default_label, dtype=np.int64)


def _as_single_mesh(scene_or_mesh: trimesh.Scene | trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=geometry.vertices, faces=geometry.faces, visual=geometry.visual)
                for geometry in scene_or_mesh.geometry.values()
            )
        )
    return scene_or_mesh


def _load_urdf_mesh(urdf_path: Path) -> trimesh.Trimesh:
    root = ET.fromstring(urdf_path.read_text(encoding="utf-8"))

    mesh_nodes = root.findall(".//collision/geometry/mesh")
    if not mesh_nodes:
        mesh_nodes = root.findall(".//visual/geometry/mesh")
    if not mesh_nodes:
        raise ValueError(f"No <mesh> geometry found in URDF: {urdf_path}")

    meshes: list[trimesh.Trimesh] = []
    for mesh_node in mesh_nodes:
        filename = mesh_node.attrib.get("filename")
        if not filename:
            continue
        mesh_path = (urdf_path.parent / filename).resolve()
        mesh_scale = mesh_node.attrib.get("scale")
        loaded = _as_single_mesh(trimesh.load(mesh_path, force="mesh"))
        if mesh_scale:
            scale_values = [float(value) for value in mesh_scale.split()]
            if len(scale_values) == 1:
                loaded.apply_scale(scale_values[0])
            elif len(scale_values) == 3:
                loaded.vertices *= np.asarray(scale_values, dtype=np.float64)
        meshes.append(loaded)

    if not meshes:
        raise ValueError(f"URDF mesh paths could not be resolved: {urdf_path}")
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(tuple(meshes))


def load_mesh_from_pose_entry(mesh_path: str | Path, scale: Any, pose: Any) -> trimesh.Trimesh:
    resolved = Path(mesh_path).expanduser().resolve()
    mesh = _load_urdf_mesh(resolved) if resolved.suffix.lower() == ".urdf" else _as_single_mesh(trimesh.load(resolved, force="mesh"))
    mesh = mesh.copy()

    scale_array = np.asarray(scale, dtype=np.float64)
    if scale_array.ndim == 0:
        mesh.apply_scale(float(scale_array))
    elif scale_array.size == 1:
        mesh.apply_scale(float(scale_array.reshape(-1)[0]))
    elif scale_array.size == 3:
        mesh.vertices *= scale_array.reshape(3)
    else:
        raise ValueError(f"Unsupported scale format for mesh pose entry: {scale!r}")

    mesh.apply_transform(np.asarray(pose, dtype=np.float64))
    return mesh


def build_scene_meshes(mesh_pose_list: list[tuple[Any, Any, Any]]) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh]]:
    mesh_list = [load_mesh_from_pose_entry(mesh_path, scale, pose) for mesh_path, scale, pose in mesh_pose_list]
    if not mesh_list:
        return trimesh.Trimesh(), []
    if len(mesh_list) == 1:
        return mesh_list[0].copy(), mesh_list
    return trimesh.util.concatenate(tuple(mesh_list)), mesh_list


def sample_instance_sdf(
    mesh_pose_list: list[tuple[Any, Any, Any]],
    num_sdf_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    scene, mesh_list = build_scene_meshes(mesh_pose_list)
    if not mesh_list or scene.is_empty:
        return np.zeros((num_sdf_points, 3), dtype=np.float32), np.zeros((0, num_sdf_points), dtype=np.float32)

    bounds = scene.bounds.astype(np.float32)
    points = np.random.rand(num_sdf_points, 3).astype(np.float32)
    points = points * (bounds[[1]] + 0.04 - bounds[[0]]) + bounds[[0]] - 0.02
    sdf = np.zeros((len(mesh_list), num_sdf_points), dtype=np.float32)
    for mesh_idx, mesh in enumerate(mesh_list):
        occ = mesh.contains(points)
        sdf[mesh_idx] = np.where(occ, 1.0, -1.0).astype(np.float32)
    return points.astype(np.float32), sdf.astype(np.float32)


def build_processed_sample(
    raw_root: Path,
    scene_id: str,
    voxel_size: float,
    num_sdf_points: int,
    num_grasp_points_scene: int,
    category_map: dict[str, int] | None = None,
) -> dict[str, Any]:
    category_map = category_map or {}

    pointcloud_data = np.load(raw_root / "pointcloud" / f"{scene_id}.npz")
    full_pointcloud_data = np.load(raw_root / "full_pointcloud" / f"{scene_id}.npz")
    mesh_pose_list = load_mesh_pose_list(raw_root / "mesh_pose_list" / f"{scene_id}.npz")

    points = pointcloud_data["pc"].astype(np.float32)
    colors = pointcloud_data["colors"].astype(np.float32) if "colors" in pointcloud_data else None
    normals = pointcloud_data["normals"].astype(np.float32) if "normals" in pointcloud_data else None

    full_points = full_pointcloud_data["pc"].astype(np.float32)
    full_instances = full_pointcloud_data["instances"].astype(np.int64)
    point_instance_ids = map_pointcloud_instances(points, full_points, full_instances)

    voxelized = quantize_points(points, colors, normals, point_instance_ids, voxel_size=voxel_size)
    unique_instances = sorted(instance_id for instance_id in np.unique(voxelized["instance_ids"]).tolist() if instance_id > 0)
    instance_id_map = {instance_id: local_idx for local_idx, instance_id in enumerate(unique_instances)}

    sdf_points, sdf = sample_instance_sdf(mesh_pose_list, num_sdf_points)
    candidate_rows = load_candidate_rows(raw_root).get(scene_id, [])
    scene_grasp_points, scene_centric_labels, object_centric_labels = build_scene_grasp_targets(
        candidate_rows,
        voxelized["raw_coordinates"],
        voxelized["instance_ids"],
        instance_id_map,
        grasp_point_count=num_grasp_points_scene,
        scene_id=scene_id,
    )

    if len(unique_instances) != sdf.shape[0]:
        min_instances = min(len(unique_instances), sdf.shape[0])
        unique_instances = unique_instances[:min_instances]
        scene_centric_labels = scene_centric_labels[:min_instances]
        object_centric_labels = object_centric_labels[:min_instances]
        sdf = sdf[:min_instances]

    sample = {
        "scene_id": scene_id,
        "quantized_coords": torch.from_numpy(voxelized["quantized_coords"]),
        "raw_coordinates": torch.from_numpy(voxelized["raw_coordinates"]),
        "colors": torch.from_numpy(voxelized["colors"]),
        "normals": torch.from_numpy(voxelized["normals"]),
        "instance_ids": torch.from_numpy(voxelized["instance_ids"]),
        "sdf_points": torch.from_numpy(sdf_points),
        "scene_grasp_points": torch.from_numpy(scene_grasp_points),
        "targets": {
            "labels": torch.from_numpy(build_semantic_labels(unique_instances, category_map)),
            "masks": torch.from_numpy(build_instance_masks(voxelized["instance_ids"], unique_instances)),
            "sdf": torch.from_numpy(sdf),
            "scene_centric_labels": torch.from_numpy(scene_centric_labels),
            "object_centric_labels": torch.from_numpy(object_centric_labels),
        },
        "meta": {
            "scene_id": scene_id,
            "num_instances": len(unique_instances),
            "num_points": int(voxelized["raw_coordinates"].shape[0]),
            "num_sdf_points": int(sdf_points.shape[0]),
            "num_grasp_points_scene": int(scene_grasp_points.shape[0]),
            "instance_ids": unique_instances,
            "status": "best-faith",
        },
    }
    return sample


def write_processed_shard(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    voxel_size: float = 0.003,
    num_sdf_points: int = 32768,
    num_grasp_points_scene: int = 128,
    val_fraction: float = 0.1,
    category_map_path: str | Path | None = None,
    reconstruction_version: str = "best-faith-v1",
    resume: bool = True,
) -> dict[str, Any]:
    raw_path = Path(raw_root).resolve()
    output_path = ensure_dir(output_root)
    samples_dir = ensure_dir(output_path / "samples")

    scene_ids = list_raw_scene_ids(raw_path)
    category_map = load_optional_category_map(category_map_path)
    split_by_scene = assign_scene_splits(scene_ids, val_fraction=val_fraction)
    generator_info = load_raw_generator_info(raw_path)

    scene_rows = []
    for scene_id in scene_ids:
        sample_path = samples_dir / f"{scene_id}.pt"
        if not (resume and sample_path.exists()):
            sample = build_processed_sample(
                raw_path,
                scene_id,
                voxel_size=voxel_size,
                num_sdf_points=num_sdf_points,
                num_grasp_points_scene=num_grasp_points_scene,
                category_map=category_map,
            )
            torch.save(sample, sample_path)
            meta = sample["meta"]
        else:
            saved = torch.load(sample_path, map_location="cpu")
            meta = saved.get("meta", {})

        scene_rows.append(
            {
                "scene_id": scene_id,
                "split": split_by_scene.get(scene_id, "train"),
                "sample_path": str(Path("samples") / f"{scene_id}.pt"),
                "num_instances": int(meta.get("num_instances", 0)),
                "num_points": int(meta.get("num_points", 0)),
                "num_sdf_points": int(meta.get("num_sdf_points", num_sdf_points)),
                "num_grasp_points_scene": int(meta.get("num_grasp_points_scene", num_grasp_points_scene)),
                "status": str(meta.get("status", "best-faith")),
            }
        )

    split_counts = Counter(row["split"] for row in scene_rows)
    manifest = {
        "format_version": "icg-reconstructed-shard-v1",
        "reconstruction_status": "best-faith",
        "reconstruction_version": reconstruction_version,
        "source_raw_root": str(raw_path),
        "generator_args": generator_info,
        "source_urdf_snapshot": snapshot_tree(OFFICIAL_URDF_ROOT / str(generator_info.get("object_set", ""))),
        "sample_schema": DEFAULT_SAMPLE_SCHEMA,
        "settings": {
            "voxel_size": voxel_size,
            "num_sdf_points": num_sdf_points,
            "num_grasp_points_scene": num_grasp_points_scene,
            "val_fraction": val_fraction,
        },
        "scene_count": len(scene_rows),
        "split_counts": dict(split_counts),
        "scenes": scene_rows,
    }

    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return manifest


def build_raw_generation_manifest(
    *,
    raw_root: Path,
    scene: str,
    object_set: str,
    urdf_root: Path,
    num_grasps: int,
    grasps_per_scene: int,
    num_proc: int,
    seed: int,
    num_views: int,
    contact_based: bool,
    sample_furthest: bool,
    num_rotations: int,
    horizontal_percentile: float,
    object_count_lambda: int,
) -> dict[str, Any]:
    return {
        "format_version": "icg-raw-generation-v1",
        "raw_root": str(raw_root),
        "scene": scene,
        "object_set": object_set,
        "urdf_root": str(urdf_root),
        "num_grasps": int(num_grasps),
        "grasps_per_scene": int(grasps_per_scene),
        "num_proc": int(num_proc),
        "seed": int(seed),
        "num_views": int(num_views),
        "contact_based": bool(contact_based),
        "sample_furthest": bool(sample_furthest),
        "num_rotations": int(num_rotations),
        "horizontal_percentile": float(horizontal_percentile),
        "object_count_lambda": int(object_count_lambda),
        "urdf_snapshot": snapshot_tree(urdf_root / object_set),
    }


def run_raw_generation(
    raw_root: str | Path,
    *,
    scene: str,
    object_set: str,
    urdf_root: str | Path = OFFICIAL_URDF_ROOT,
    num_grasps: int,
    grasps_per_scene: int = 120,
    num_proc: int = 1,
    seed: int = 0,
    num_views: int = 1,
    contact_based: bool = True,
    sample_furthest: bool = True,
    num_rotations: int = 12,
    horizontal_percentile: float = 0.85,
    object_count_lambda: int = 5,
    resume: bool = True,
) -> dict[str, Any]:
    raw_path = ensure_dir(raw_root)
    urdf_path = Path(urdf_root).resolve()
    manifest = build_raw_generation_manifest(
        raw_root=raw_path,
        scene=scene,
        object_set=object_set,
        urdf_root=urdf_path,
        num_grasps=num_grasps,
        grasps_per_scene=grasps_per_scene,
        num_proc=num_proc,
        seed=seed,
        num_views=num_views,
        contact_based=contact_based,
        sample_furthest=sample_furthest,
        num_rotations=num_rotations,
        horizontal_percentile=horizontal_percentile,
        object_count_lambda=object_count_lambda,
    )
    manifest_path = raw_path / "raw_manifest.json"

    if resume and manifest_path.exists() and list_raw_scene_ids(raw_path):
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    command = [
        sys.executable,
        str(RAW_GENERATOR),
        str(raw_path),
        "--scene",
        scene,
        "--object-set",
        object_set,
        "--num-grasps",
        str(num_grasps),
        "--grasps-per-scene",
        str(grasps_per_scene),
        "--num-proc",
        str(num_proc),
        "--seed",
        str(seed),
        "--num-views",
        str(num_views),
        "--object-count-lambda",
        str(object_count_lambda),
        "--num-rotations",
        str(num_rotations),
        "--horizontal-percentile",
        str(horizontal_percentile),
        "--urdf-root",
        str(urdf_path),
    ]
    if contact_based:
        command.append("--contact-based")
    if sample_furthest:
        command.append("--sample-furthest")

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{DATA_COLLECTION_SRC}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(DATA_COLLECTION_SRC)
    )
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def missing_module(module_name: str) -> bool:
    try:
        __import__(module_name)
        return False
    except Exception:
        return True


def write_reconstruction_audit(
    output_dir: str | Path,
    *,
    config_path: str | Path = OFFICIAL_CONFIG,
) -> dict[str, Path]:
    output_path = ensure_dir(output_dir)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    findings = {
        "status": "best-faith",
        "official_config": str(Path(config_path).resolve()),
        "published_raw_generator": str(RAW_GENERATOR.resolve()),
        "missing_training_dataset_module": missing_module("icg_net.data.giga_contact_dataset"),
        "missing_training_collator_module": missing_module("datasets.utils"),
        "official_num_classes": config.get("loss", {}).get("num_classes"),
        "official_grasp_loss_type": config.get("loss", {}).get("grasp_loss_type"),
        "official_sdf_loss_type": config.get("loss", {}).get("sdf_loss_type"),
        "official_train_dirs": config.get("data", {}).get("train_dataset", {}).get("data_dir", []),
        "unresolved_items": [
            "Published repo contains raw synthetic generation and occupancy helpers, but not the original GigaContactDataset implementation.",
            "Published repo does not include datasets.utils.VoxelizeCollateSDF.",
            "Published config uses num_classes=7, but the class taxonomy is not documented in the published code/configs.",
            "Published training loss weights are not exposed by a trainer module in this checkout.",
        ],
    }

    md_path = output_path / "reconstruction_audit.md"
    md_path.write_text(
        "\n".join(
            [
                "# ICG-Net Reconstruction Audit",
                "",
                f"- Status: `{findings['status']}`",
                f"- Official config: `{findings['official_config']}`",
                f"- Raw generator present: `{findings['published_raw_generator']}`",
                f"- Missing `GigaContactDataset`: `{findings['missing_training_dataset_module']}`",
                f"- Missing `VoxelizeCollateSDF`: `{findings['missing_training_collator_module']}`",
                f"- Official `num_classes`: `{findings['official_num_classes']}`",
                f"- Official `grasp_loss_type`: `{findings['official_grasp_loss_type']}`",
                f"- Official `sdf_loss_type`: `{findings['official_sdf_loss_type']}`",
                "",
                "## Unresolved Items",
                "",
                *[f"- {item}" for item in findings["unresolved_items"]],
                "",
                "## Verification Template",
                "",
                "Suggested issue/email topic:",
                "",
                "`ICG-Net training data reconstruction: class taxonomy, processed shard schema, and training loss weights`",
                "",
                "Do not call the rebuilt corpus paper-exact until these items are confirmed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    json_path = output_path / "reconstruction_audit.json"
    json_path.write_text(json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8")

    return {"markdown": md_path, "json": json_path}


def discover_object_packages(source_root: str | Path) -> list[Path]:
    root = Path(source_root).resolve()
    packages: list[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and list(child.rglob("*.urdf")):
            packages.append(child)
        elif child.is_file() and child.suffix == ".urdf":
            packages.append(child)
    return packages


def compute_object_split_counts(count: int) -> dict[str, int]:
    if count < 3:
        raise ValueError("Extension object catalogs require at least 3 custom object packages.")
    train = max(1, round(count * 0.7))
    val = max(1, round(count * 0.15))
    test = count - train - val
    if test < 1:
        test = 1
        if train > val:
            train -= 1
        else:
            val -= 1
    return {"train": train, "val": val, "test": test}


def register_extension_objects(
    source_root: str | Path,
    *,
    catalog_root: str | Path | None = None,
    scene_scopes: Sequence[str] = ("packed", "pile"),
    symlink: bool = True,
) -> dict[str, Any]:
    packages = discover_object_packages(source_root)
    counts = compute_object_split_counts(len(packages))
    catalog_path = ensure_dir(resolve_icg_object_root(catalog_root))

    ordered = sorted(packages, key=lambda path: path.name.lower())
    split_names = (
        ["train"] * counts["train"] + ["val"] * counts["val"] + ["test"] * counts["test"]
    )
    assignments = list(zip(ordered, split_names))

    manifest_entries = []
    for scene in scene_scopes:
        scene_root = ensure_dir(catalog_path / f"{scene}_ext")
        for split in ("train", "val", "test"):
            ensure_dir(scene_root / split)

    for package, split in assignments:
        for scene in scene_scopes:
            destination = catalog_path / f"{scene}_ext" / split / package.name
            if destination.exists():
                if destination.is_symlink() or destination.is_file():
                    destination.unlink()
                else:
                    shutil.rmtree(destination)

            if symlink:
                destination.symlink_to(package, target_is_directory=package.is_dir())
            elif package.is_dir():
                shutil.copytree(package, destination)
            else:
                shutil.copy2(package, destination)

        manifest_entries.append({"package": package.name, "source": str(package), "split": split})

    manifest = {
        "format_version": "icg-extension-object-catalog-v1",
        "catalog_root": str(catalog_path),
        "scene_scopes": list(scene_scopes),
        "entries": manifest_entries,
    }
    manifest_path = catalog_path / "catalog_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


class ReconstructedICGDataset(Dataset):
    """Best-faith replacement for the unpublished GigaContactDataset."""

    def __init__(
        self,
        data_dir: str | Sequence[str],
        mode: str = "train",
        data_percent: float = 1.0,
        num_sdf_points: int = 32768,
        num_grasp_points_scene: int = 128,
        add_colors: bool = False,
        add_normals: bool = False,
        add_z_coordinate: bool = False,
        load_grasps: bool = True,
        balance_grasps: bool = False,
        positive_grasp_fraction: float = 0.5,
        seed: int = 0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_sdf_points = num_sdf_points
        self.num_grasp_points_scene = num_grasp_points_scene
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_z_coordinate = add_z_coordinate
        self.load_grasps = load_grasps
        self.balance_grasps = balance_grasps
        self.positive_grasp_fraction = positive_grasp_fraction
        self.seed = seed

        roots = [Path(path).resolve() for path in ([data_dir] if isinstance(data_dir, (str, os.PathLike)) else data_dir)]
        self.samples: list[dict[str, Any]] = []
        split_name = {"validation": "val", "val": "val", "train": "train", "test": "test"}.get(mode, mode)

        for root in roots:
            manifest_path = root / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Processed shard manifest not found: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for row in manifest.get("scenes", []):
                if split_name in {"train", "val", "test"} and row.get("split") != split_name:
                    continue
                self.samples.append(
                    {
                        "root": root,
                        "scene_id": row["scene_id"],
                        "sample_path": root / row["sample_path"],
                    }
                )

        self.samples.sort(key=lambda item: (item["scene_id"], str(item["sample_path"])))
        if 0 < data_percent < 1.0 and self.samples:
            keep = max(1, round(len(self.samples) * data_percent))
            ordered = sorted(self.samples, key=lambda item: hash_scene_id(item["scene_id"]))
            self.samples = ordered[:keep]

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, total: int, target: int, index: int) -> np.ndarray:
        if total <= target:
            return np.arange(total)
        generator = np.random.default_rng(self.seed + index)
        return np.sort(generator.choice(np.arange(total), size=target, replace=False))

    def _sample_grasp_indices(self, labels: torch.Tensor, target: int, index: int) -> np.ndarray:
        total = labels.shape[1]
        if total <= target:
            return np.arange(total)
        if not self.balance_grasps:
            return self._sample_indices(total, target, index)

        generator = np.random.default_rng(self.seed + index)
        positive_mask = labels[..., :12].amax(dim=(0, 2)) > 0
        positive = np.flatnonzero(positive_mask.numpy())
        negative = np.flatnonzero((~positive_mask).numpy())
        if len(positive) == 0 or len(negative) == 0:
            return self._sample_indices(total, target, index)

        positive_target = min(len(positive), max(1, round(target * self.positive_grasp_fraction)))
        negative_target = min(len(negative), target - positive_target)
        if positive_target + negative_target < target:
            positive_target = min(len(positive), positive_target + target - positive_target - negative_target)
        if positive_target + negative_target < target:
            negative_target = min(len(negative), target - positive_target)

        selected = np.concatenate(
            [
                generator.choice(positive, size=positive_target, replace=False),
                generator.choice(negative, size=negative_target, replace=False),
            ]
        )
        return np.sort(selected)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.samples[index]
        sample = torch.load(record["sample_path"], map_location="cpu")

        raw_coordinates = sample["raw_coordinates"].float()
        colors = sample["colors"].float()
        normals = sample["normals"].float()

        features = [torch.ones((raw_coordinates.shape[0], 1), dtype=torch.float32)]
        if self.add_colors:
            features.append(colors)
        if self.add_normals:
            features.append(normals)
        if self.add_z_coordinate:
            features.append(raw_coordinates[:, 2:3])

        sdf_points = sample["sdf_points"].float()
        sdf = sample["targets"]["sdf"].float()
        if self.num_sdf_points > 0:
            sdf_indices = self._sample_indices(sdf_points.shape[0], min(self.num_sdf_points, sdf_points.shape[0]), index)
            sdf_points = sdf_points[sdf_indices]
            sdf = sdf[:, sdf_indices]

        scene_grasp_points = sample["scene_grasp_points"].float()
        scene_centric_labels = sample["targets"]["scene_centric_labels"].float()
        object_centric_labels = sample["targets"]["object_centric_labels"].float()
        if self.load_grasps and self.num_grasp_points_scene > 0:
            grasp_indices = self._sample_grasp_indices(
                scene_centric_labels,
                min(self.num_grasp_points_scene, scene_grasp_points.shape[0]),
                index + 7919,
            )
            scene_grasp_points = scene_grasp_points[grasp_indices]
            scene_centric_labels = scene_centric_labels[:, grasp_indices]
            object_centric_labels = object_centric_labels[:, grasp_indices]
        else:
            scene_grasp_points = torch.zeros((0, 3), dtype=torch.float32)
            scene_centric_labels = torch.zeros((scene_centric_labels.shape[0], 0, 13), dtype=torch.float32)
            object_centric_labels = torch.zeros((object_centric_labels.shape[0], 0, 13), dtype=torch.float32)

        targets = {
            "labels": sample["targets"]["labels"].long(),
            "masks": sample["targets"]["masks"].float(),
            "sdf": sdf,
            "scene_centric_labels": scene_centric_labels,
            "object_centric_labels": object_centric_labels,
        }

        return {
            "quantized_coords": sample["quantized_coords"].int(),
            "raw_coordinates": raw_coordinates,
            "input_points": raw_coordinates,
            "features": torch.cat(features, dim=-1).float(),
            "sdf_points": sdf_points,
            "scene_grasp_points": scene_grasp_points,
            "targets": targets,
            "mask_type": "masks",
            "meta": sample.get("meta", {"scene_id": record["scene_id"]}),
        }


class ReconstructedVoxelizeCollateSDF:
    """Best-faith replacement for the unpublished VoxelizeCollateSDF."""

    def __init__(self, voxel_size: float = 0.003, mode: str = "train", **_: Any) -> None:
        self.voxel_size = voxel_size
        self.mode = mode

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        result = {
            "raw_coordinates": [item["raw_coordinates"] for item in batch],
            "input_points": [item["input_points"] for item in batch],
            "sdf_points": [item["sdf_points"] for item in batch],
            "scene_grasp_points": [item["scene_grasp_points"] for item in batch],
            "targets": [item["targets"] for item in batch],
            "mask_type": batch[0]["mask_type"],
            "meta": [item.get("meta", {}) for item in batch],
            "coords": [item["raw_coordinates"] for item in batch],
            "features": [item["features"] for item in batch],
        }

        try:
            import MinkowskiEngine as ME

            coordinates = [item["quantized_coords"].int() for item in batch]
            features = [item["features"].float() for item in batch]
            result["voxelized_data"] = ME.SparseTensor(
                torch.cat(features, dim=0),
                ME.utils.batched_coordinates(coordinates),
            )
        except Exception:
            result["voxelized_data"] = None

        return result
