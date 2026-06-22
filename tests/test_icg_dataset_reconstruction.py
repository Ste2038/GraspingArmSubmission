from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts import icg_dataset_lib
from scripts.audit_icg_reconstructed_targets import run_audit
from scripts.build_icg_dataset import canonical_shard_seed, select_shard_names


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "third_party/icg_benchmark/data/icgnet/51--0.656/config.yaml"


def _write_raw_fixture(raw_root: Path) -> tuple[Path, list[str]]:
    (raw_root / "scenes").mkdir(parents=True, exist_ok=True)
    (raw_root / "pointcloud").mkdir(parents=True, exist_ok=True)
    (raw_root / "full_pointcloud").mkdir(parents=True, exist_ok=True)
    (raw_root / "mesh_pose_list").mkdir(parents=True, exist_ok=True)

    info = {
        "object_set": "fixture/train",
        "scene": "packed",
        "grasps_per_scene": 4,
        "num_views": 1,
    }
    (raw_root / "info.yaml").write_text(yaml.safe_dump(info), encoding="utf-8")

    mesh_path = raw_root / "fixture_mesh.obj"
    mesh_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scene_ids = ["scene_alpha", "scene_beta"]
    centers = [0.10, 0.24]
    for scene_id, center in zip(scene_ids, centers):
        pointcloud = np.array(
            [
                [center, 0.10, 0.07],
                [center + 0.01, 0.11, 0.08],
                [center - 0.01, 0.09, 0.075],
                [center + 0.005, 0.10, 0.082],
            ],
            dtype=np.float32,
        )
        full_pointcloud = np.array(
            [
                [center, 0.10, 0.07],
                [center + 0.01, 0.11, 0.08],
                [center - 0.01, 0.09, 0.075],
                [center + 0.005, 0.10, 0.082],
                [center + 0.008, 0.095, 0.078],
            ],
            dtype=np.float32,
        )
        colors = np.tile(np.array([[0.2, 0.4, 0.6]], dtype=np.float32), (pointcloud.shape[0], 1))
        full_colors = np.tile(np.array([[0.3, 0.5, 0.7]], dtype=np.float32), (full_pointcloud.shape[0], 1))
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (pointcloud.shape[0], 1))
        full_normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (full_pointcloud.shape[0], 1))
        instances = np.ones((full_pointcloud.shape[0],), dtype=np.int64)
        mesh_pose_list = np.array([(str(mesh_path), 1.0, np.eye(4, dtype=np.float32))], dtype=object)

        np.savez_compressed(raw_root / "scenes" / f"{scene_id}.npz", depth_imgs=np.zeros((1, 2, 2)), extrinsics=np.zeros((1, 7)))
        np.savez_compressed(
            raw_root / "pointcloud" / f"{scene_id}.npz",
            pc=pointcloud,
            colors=colors,
            normals=normals,
        )
        np.savez_compressed(
            raw_root / "full_pointcloud" / f"{scene_id}.npz",
            pc=full_pointcloud,
            colors=full_colors,
            normals=full_normals,
            instances=instances,
        )
        np.savez_compressed(raw_root / "mesh_pose_list" / f"{scene_id}.npz", pc=mesh_pose_list)

    candidate_path = raw_root / "grasps_candidate.csv"
    fieldnames = ["scene_id", "x", "y", "z", "width", "target"] + [f"label_{idx}" for idx in range(12)]
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scene_id, center in zip(scene_ids, centers):
            row = {"scene_id": scene_id, "x": center, "y": 0.10, "z": 0.07, "width": 0.045, "target": 1}
            for idx in range(12):
                row[f"label_{idx}"] = 1.0 if idx == 0 else 0.0
            writer.writerow(row)

    return raw_root, scene_ids


def _patch_sdf_sampler(monkeypatch) -> None:
    def fake_sample_instance_sdf(mesh_pose_list, num_sdf_points):
        points = np.linspace(0.0, 0.09, num_sdf_points * 3, dtype=np.float32).reshape(num_sdf_points, 3)
        sdf = np.ones((len(mesh_pose_list), num_sdf_points), dtype=np.float32)
        sdf[:, ::2] = -1.0
        return points, sdf

    monkeypatch.setattr(icg_dataset_lib, "sample_instance_sdf", fake_sample_instance_sdf)


def test_write_processed_shard_is_resumable_and_schema_matches(tmp_path: Path, monkeypatch) -> None:
    _patch_sdf_sampler(monkeypatch)
    raw_root, scene_ids = _write_raw_fixture(tmp_path / "raw")
    output_root = tmp_path / "processed"

    manifest = icg_dataset_lib.write_processed_shard(
        raw_root,
        output_root,
        num_sdf_points=16,
        num_grasp_points_scene=8,
        val_fraction=0.5,
    )
    manifest_again = icg_dataset_lib.write_processed_shard(
        raw_root,
        output_root,
        num_sdf_points=16,
        num_grasp_points_scene=8,
        val_fraction=0.5,
        resume=True,
    )

    assert manifest["scene_count"] == len(scene_ids)
    assert manifest["split_counts"] == manifest_again["split_counts"]
    assert sorted(row["scene_id"] for row in manifest["scenes"]) == scene_ids
    assert set(row["split"] for row in manifest["scenes"]) == {"train", "val"}

    sample_path = output_root / manifest["scenes"][0]["sample_path"]
    payload = torch.load(sample_path, map_location="cpu")
    assert set(payload["targets"].keys()) == {
        "labels",
        "masks",
        "sdf",
        "scene_centric_labels",
        "object_centric_labels",
    }
    assert payload["scene_grasp_points"].shape[-1] == 3
    assert payload["targets"]["scene_centric_labels"].shape[-1] == 13


def test_register_extension_objects_creates_split_catalog(tmp_path: Path) -> None:
    source_root = tmp_path / "objects_in"
    source_root.mkdir()
    for name in ("alpha", "beta", "gamma", "delta"):
        package_dir = source_root / name
        package_dir.mkdir()
        (package_dir / f"{name}.urdf").write_text(f"<robot name='{name}'></robot>\n", encoding="utf-8")

    manifest = icg_dataset_lib.register_extension_objects(
        source_root,
        catalog_root=tmp_path / "catalog",
        symlink=False,
    )

    assert len(manifest["entries"]) == 4
    for scene in ("packed", "pile"):
        for split in ("train", "val", "test"):
            split_dir = tmp_path / "catalog" / f"{scene}_ext" / split
            assert split_dir.exists()
    assigned_splits = {entry["split"] for entry in manifest["entries"]}
    assert assigned_splits == {"train", "val", "test"}


def test_reconstructed_dataset_balances_grasp_point_sampling(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()
    (processed_root / "manifest.json").write_text(
        '{"scenes": []}',
        encoding="utf-8",
    )
    dataset = icg_dataset_lib.ReconstructedICGDataset(
        processed_root,
        balance_grasps=True,
        positive_grasp_fraction=0.5,
        seed=3,
    )
    labels = torch.zeros((1, 10, 13), dtype=torch.float32)
    labels[:, [0, 1, 2], 0] = 1.0

    selected = dataset._sample_grasp_indices(labels, target=6, index=0)

    positives = {0, 1, 2}
    positive_count = sum(int(index in positives) for index in selected)
    assert len(selected) == 6
    assert positive_count == 3


def test_canonical_shard_seed_uses_absolute_selected_index() -> None:
    shard_names = [f"packed_2M_contact_{index}" for index in range(5)]
    selected = select_shard_names(shard_names, [2, 3], "packed")

    assert selected == ["packed_2M_contact_2", "packed_2M_contact_3"]
    assert canonical_shard_seed(0, "packed", selected[0]) == 202
    assert canonical_shard_seed(0, "packed", selected[1]) == 303
    assert canonical_shard_seed(0, "pile", "pile_2M_contact_2") == 10202


def test_reconstructed_target_audit_writes_reports(tmp_path: Path, monkeypatch) -> None:
    _patch_sdf_sampler(monkeypatch)
    raw_root, _ = _write_raw_fixture(tmp_path / "raw_audit")
    processed_root = tmp_path / "processed_audit"
    icg_dataset_lib.write_processed_shard(
        raw_root,
        processed_root,
        num_sdf_points=16,
        num_grasp_points_scene=8,
        val_fraction=0.5,
    )

    outputs = run_audit(processed_root, tmp_path / "target_audit")

    assert outputs["json"].exists()
    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["summary"]["all"][0]["samples"] == 2
    assert payload["summary"]["all"][0]["zero_width_positive_points"] == 0
    assert "best-faith" in outputs["markdown"].read_text(encoding="utf-8")


def test_reconstructed_dataset_smoke_training_pipeline(tmp_path: Path, monkeypatch) -> None:
    _patch_sdf_sampler(monkeypatch)
    raw_root, _ = _write_raw_fixture(tmp_path / "raw_train")
    processed_root = tmp_path / "processed_train"
    icg_dataset_lib.write_processed_shard(
        raw_root,
        processed_root,
        num_sdf_points=16,
        num_grasp_points_scene=8,
        val_fraction=0.5,
    )

    overlay = {
        "wrapper": {
            "run_name": "pytest_reconstructed_smoke",
            "max_epochs": 1,
            "loss_weights": {
                "loss_ce": 1.0,
                "loss_mask": 1.0,
                "loss_dice": 1.0,
                "loss_sdf": 1.0,
                "loss_grasp_scene": 1.0,
                "loss_grasp_width_scene": 1.0,
            },
            "adapter": {"_target_": "scripts.icg_training_lib.SyntheticSmokeAdapter"},
            "train_dataset": {
                "_target_": "scripts.icg_dataset_lib.ReconstructedICGDataset",
                "data_dir": [str(processed_root)],
                "mode": "train",
                "num_sdf_points": 16,
                "num_grasp_points_scene": 8,
            },
            "validation_dataset": {
                "_target_": "scripts.icg_dataset_lib.ReconstructedICGDataset",
                "data_dir": [str(processed_root)],
                "mode": "val",
                "num_sdf_points": 16,
                "num_grasp_points_scene": 8,
            },
            "train_collate": {
                "_target_": "scripts.icg_dataset_lib.ReconstructedVoxelizeCollateSDF",
                "voxel_size": 0.003,
                "mode": "train",
            },
            "validation_collate": {
                "_target_": "scripts.icg_dataset_lib.ReconstructedVoxelizeCollateSDF",
                "voxel_size": 0.003,
                "mode": "val",
            },
            "train_dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 0,
            },
            "validation_dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": 1,
                "shuffle": False,
                "num_workers": 0,
            },
        },
        "model": {
            "_target_": "scripts.icg_training_lib.SyntheticICGNetLikeModel",
            "num_queries": 4,
            "num_classes": 7,
            "hidden_dim": 32,
        },
        "loss": {
            "_target_": "icg_net.trainer.criterion.criterion.SetCriterionWithWidth",
            "class_weights": -1,
            "eos_coef": 0.1,
            "exclude_sdf_class": [],
            "grasp_loss_type": "bce",
            "losses": ["labels", "masks", "sdf", "grasps"],
            "num_classes": 7,
            "num_points": -1,
            "sdf_clip": 0.05,
            "sdf_loss_type": "bce",
        },
        "matcher": {
            "_target_": "icg_net.trainer.matcher.instance_matcher.HungarianMatcher",
            "cost_class": 1.0,
            "cost_dice": 1.0,
            "cost_mask": 1.0,
            "cost_pose": 0.0,
            "costs": {
                "grasp_scene": 1.0,
                "grasp_width_scene": 1.0,
                "scene_occ": 1.0,
                "sdf": 1.0,
                "sdf_matching": 0.0,
            },
            "num_points": -1,
        },
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 0.001},
        "trainer": {"max_epochs": 1},
    }
    overlay_path = tmp_path / "reconstructed_smoke.yaml"
    overlay_path.write_text(yaml.safe_dump(overlay), encoding="utf-8")

    output_root = tmp_path / "training_logs"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/train_icg.py"),
            "--config",
            str(BASE_CONFIG),
            "--overlay",
            str(overlay_path),
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--max-epochs",
            "1",
            "--run-name",
            "pytest_reconstructed",
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    run_dirs = sorted(output_root.glob("pytest_reconstructed_*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    assert (run_dir / "metrics.jsonl").exists()

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/plot_icg_training_metrics.py"),
            str(run_dir),
            "--output-dir",
            str(run_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert (run_dir / "loss_components.png").exists()
    assert (run_dir / "val_scene_grasp_f1.png").exists()
