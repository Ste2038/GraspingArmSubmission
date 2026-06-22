"""Training helpers, synthetic smoke components, and plotting utilities for ICG-Net."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


TARGET_REWRITES = {
    "icg_net.matcher.instance_matcher.HungarianMatcher": "icg_net.trainer.matcher.instance_matcher.HungarianMatcher",
    "icg_net.criterion.criterion.SetCriterion": "icg_net.trainer.criterion.criterion.SetCriterion",
    "icg_net.criterion.criterion.SetCriterionWithWidth": "icg_net.trainer.criterion.criterion.SetCriterionWithWidth",
    "icg_net.criterion.criterion.SetCriterionGIGA": "icg_net.trainer.criterion.criterion.SetCriterionGIGA",
    "icg_net.criterion.criterion.SetCriterionAcronym": "icg_net.trainer.criterion.criterion.SetCriterionAcronym",
    "icg_net.optim.cosine_ann.LinearWarmupCosineAnnealingLR": (
        "icg_net.trainer.optim.cosine_ann.LinearWarmupCosineAnnealingLR"
    ),
    "icg_net.data.giga_contact_dataset.GigaContactDataset": "scripts.icg_dataset_lib.ReconstructedICGDataset",
    "icg_net.data.giga_contact_dataset.GigaContactTestDataset": "scripts.icg_dataset_lib.ReconstructedICGDataset",
    "datasets.utils.VoxelizeCollateSDF": "scripts.icg_dataset_lib.ReconstructedVoxelizeCollateSDF",
}

PAPER_TERM_ROWS = [
    {
        "section": "training_term",
        "paper_term": "Instance segmentation loss",
        "code_keys": "loss_mask + loss_dice",
        "paper_target": "Not reported",
        "notes": "The paper describes the panoptic/instance term structurally; use the raw BCE-like mask term plus DICE.",
    },
    {
        "section": "training_term",
        "paper_term": "Semantic/class term",
        "code_keys": "loss_ce",
        "paper_target": "Not reported",
        "notes": "The code uses cross-entropy over instance/query classes; this is not a separate semantic BCE term.",
    },
    {
        "section": "training_term",
        "paper_term": "Grasp loss",
        "code_keys": "loss_grasp_scene + loss_grasp_width_scene",
        "paper_target": "Not reported",
        "notes": (
            "The paper describes direction classification plus width L2. "
            "The official checkpoint config uses SetCriterionWithWidth with BCE affordance supervision plus width MSE."
        ),
    },
    {
        "section": "training_term",
        "paper_term": "3D reconstruction occupancy loss",
        "code_keys": "loss_sdf (when sdf_loss_type == bce)",
        "paper_target": "Not reported",
        "notes": "loss_scene_occ is treated as an auxiliary scene-level occupancy term, not as the main paper loss.",
    },
    {
        "section": "training_term",
        "paper_term": "Main validation metric",
        "code_keys": "val/scene_grasp/f1",
        "paper_target": "Not reported",
        "notes": "The paper states that affordance F1 on the validation set is used for early stopping, but no numeric target is published.",
    },
]

PAPER_BENCHMARK_ROWS = [
    {"scene": "packed", "metric": "GSR", "paper_value": "97.7 +/- 0.9"},
    {"scene": "packed", "metric": "DR", "paper_value": "97.5 +/- 0.3"},
    {"scene": "pile", "metric": "GSR", "paper_value": "92.0 +/- 2.6"},
    {"scene": "pile", "metric": "DR", "paper_value": "94.1 +/- 1.4"},
]


def load_yaml_config(path: str | Path) -> DictConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return OmegaConf.create(yaml.safe_load(handle))


def merge_configs(base_path: str | Path, overlay_paths: Iterable[str | Path] | None = None) -> DictConfig:
    cfg = load_yaml_config(base_path)
    rewrite_target_paths(cfg)
    for overlay in overlay_paths or []:
        overlay_cfg = load_yaml_config(overlay)
        rewrite_target_paths(overlay_cfg)
        cfg = OmegaConf.merge(cfg, overlay_cfg)
    return cfg


def rewrite_target_paths(node: Any) -> None:
    if isinstance(node, DictConfig):
        for key in list(node.keys()):
            value = node[key]
            if key == "_target_" and isinstance(value, str):
                node[key] = TARGET_REWRITES.get(value, value)
            else:
                rewrite_target_paths(value)
        return

    if isinstance(node, ListConfig):
        for value in node:
            rewrite_target_paths(value)


def omega_to_dict(node: Any) -> Any:
    if isinstance(node, (DictConfig, ListConfig)):
        return OmegaConf.to_container(node, resolve=True)
    return node


def scalarize(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().mean().cpu().item())
    return float(value)


def average_metric_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if value is None or math.isnan(value):
                continue
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if hasattr(obj, "to") and callable(obj.to):
        try:
            return obj.to(device)
        except TypeError:
            return obj
    return obj


def ensure_loss_weights(cfg: DictConfig) -> dict[str, float]:
    weight_cfg = cfg.get("wrapper", {}).get("loss_weights") if "wrapper" in cfg else None
    weights = omega_to_dict(weight_cfg) if weight_cfg is not None else None
    if not isinstance(weights, dict) or not weights:
        raise ValueError(
            "Missing explicit wrapper.loss_weights in the overlay config. "
            "The local wrapper does not invent hidden training weights."
        )
    return {str(key): float(value) for key, value in weights.items()}


def weighted_loss_sum(losses: dict[str, torch.Tensor], loss_weights: dict[str, float]) -> tuple[torch.Tensor, list[str]]:
    used_keys: list[str] = []
    total: torch.Tensor | None = None
    for key, weight in loss_weights.items():
        if key not in losses:
            continue
        term = losses[key] * float(weight)
        total = term if total is None else total + term
        used_keys.append(key)
    if total is None:
        raise ValueError(
            "No criterion losses matched the explicit loss_weights. "
            "Provide exact keys such as loss_mask, loss_dice, loss_ce, loss_sdf, loss_grasp_scene, loss_grasp_width_scene."
        )
    return total, used_keys


def torch_load_cpu_or_device(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def replace_key_prefix(state_dict: dict[str, torch.Tensor], old_prefix: str, new_prefix: str) -> None:
    renamed = {}
    for key in list(state_dict.keys()):
        if key.startswith(old_prefix):
            renamed[key] = new_prefix + key[len(old_prefix) :]
    for old_key, new_key in renamed.items():
        state_dict[new_key] = state_dict.pop(old_key)


def compatible_model_state_dict(
    checkpoint_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    state_dict = dict(checkpoint_state)
    replace_key_prefix(state_dict, "model.", "")
    replace_key_prefix(state_dict, "backbone.", "feature_volume.sparse_feature_extractor.")
    replace_key_prefix(state_dict, "dense_extractor.", "feature_volume.dense_extractor.")
    replace_key_prefix(state_dict, "dense_graps_interpolations", "grasp_feature_interpolator.dense_interpolations")
    replace_key_prefix(state_dict, "point_interpolation_grasp", "grasp_feature_interpolator.point_interpolations")
    replace_key_prefix(state_dict, "grasp_out_mlp", "grasp_feature_interpolator.out_mlp")
    replace_key_prefix(state_dict, "dense_occ_interpolations", "occupancy_feature_interpolator.dense_interpolations")
    replace_key_prefix(state_dict, "point_interpolation_occ", "occupancy_feature_interpolator.point_interpolations")
    replace_key_prefix(state_dict, "occ_out_mlp", "occupancy_feature_interpolator.out_mlp")

    model_state = model.state_dict()
    compatible = {}
    skipped_shape = 0
    excessive = 0
    for key, value in state_dict.items():
        if key not in model_state:
            excessive += 1
            continue
        if value.shape != model_state[key].shape:
            skipped_shape += 1
            continue
        compatible[key] = value

    missing = 0
    for key, value in model_state.items():
        if key not in compatible:
            missing += 1
            compatible[key] = value

    return compatible, {
        "loaded": len(compatible) - missing,
        "missing_filled_from_init": missing,
        "excessive_ignored": excessive,
        "shape_mismatch_ignored": skipped_shape,
    }


def load_checkpoint_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch_load_cpu_or_device(path, device)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"No state_dict/model_state_dict found in {path}")
    return state_dict


def initialize_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, int | str]:
    path = Path(checkpoint_path).expanduser()
    state_dict = load_checkpoint_state_dict(path, device)
    compatible_state, report = compatible_model_state_dict(state_dict, model)
    model.load_state_dict(compatible_state)
    return {"checkpoint": str(path.resolve()), **report}


def build_alias_metrics(prefix: str, losses: dict[str, float], sdf_loss_type: str | None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if "loss_mask" in losses:
        metrics[f"{prefix}/loss_inst_bce"] = losses["loss_mask"]
    if "loss_dice" in losses:
        metrics[f"{prefix}/loss_inst_dice"] = losses["loss_dice"]
    if "loss_mask" in losses and "loss_dice" in losses:
        metrics[f"{prefix}/loss_inst_total"] = losses["loss_mask"] + losses["loss_dice"]
    if "loss_ce" in losses:
        metrics[f"{prefix}/loss_sem_ce"] = losses["loss_ce"]
    if "loss_grasp_scene" in losses:
        metrics[f"{prefix}/loss_grasp_aff_scene"] = losses["loss_grasp_scene"]
    if "loss_grasp_width_scene" in losses:
        metrics[f"{prefix}/loss_grasp_width_scene"] = losses["loss_grasp_width_scene"]
    if "loss_grasp_scene" in losses and "loss_grasp_width_scene" in losses:
        metrics[f"{prefix}/loss_grasp_total"] = losses["loss_grasp_scene"] + losses["loss_grasp_width_scene"]
    if sdf_loss_type == "bce" and "loss_sdf" in losses:
        metrics[f"{prefix}/loss_occ_bce"] = losses["loss_sdf"]
    if "loss_scene_occ" in losses:
        metrics[f"{prefix}/loss_occ_scene_aux"] = losses["loss_scene_occ"]
    if "loss_normals" in losses:
        metrics[f"{prefix}/loss_normals"] = losses["loss_normals"]
    if "loss_total_weighted" in losses:
        metrics[f"{prefix}/loss_total_weighted"] = losses["loss_total_weighted"]
    return metrics


def as_percent_string(value: float | int | None) -> str:
    if value is None:
        return "-"
    number = float(value)
    if abs(number) <= 1.0:
        number *= 100.0
    return f"{number:.2f}"


def metric_path_from_source(source: str | Path) -> Path:
    path = Path(source)
    if path.is_dir():
        jsonl_path = path / "metrics.jsonl"
        csv_path = path / "metrics.csv"
        if jsonl_path.exists():
            return jsonl_path
        if csv_path.exists():
            return csv_path
        raise FileNotFoundError(f"No metrics.jsonl or metrics.csv found in {path}")
    return path


def load_metric_records(source: str | Path) -> list[dict[str, Any]]:
    path = metric_path_from_source(source)
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                parsed: dict[str, Any] = {}
                for key, value in row.items():
                    if value in (None, "", "nan"):
                        parsed[key] = value
                        continue
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
                rows.append(parsed)
            return rows

    raise ValueError(f"Unsupported metrics format: {path.suffix}")


def write_metric_records(records: list[dict[str, Any]], run_dir: str | Path) -> None:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_path / "metrics.jsonl"
    csv_path = run_path / "metrics.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    fieldnames = sorted({key for record in records for key in record.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def plot_training_losses(records: list[dict[str, Any]], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_rows = sorted(
        [row for row in records if row.get("split") == "train"],
        key=lambda row: (row.get("epoch", 0), row.get("global_step", 0)),
    )

    series = [
        ("train/loss_inst_bce", "Instance BCE"),
        ("train/loss_inst_dice", "Instance DICE"),
        ("train/loss_sem_ce", "Semantic/Class CE"),
        ("train/loss_grasp_aff_scene", "Grasp Affordance"),
        ("train/loss_grasp_width_scene", "Grasp Width"),
        ("train/loss_occ_bce", "Occupancy BCE"),
        ("train/loss_occ_scene_aux", "Scene Occupancy Aux"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    for key, label in series:
        xs = [row["epoch"] for row in train_rows if key in row and row[key] not in ("", None)]
        ys = [row[key] for row in train_rows if key in row and row[key] not in ("", None)]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)
        plotted = True

    ax.set_title("ICG-Net Training Loss Components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No training loss series found.", ha="center", va="center", transform=ax.transAxes)

    figure_path = output_path / "loss_components.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def plot_validation_f1(records: list[dict[str, Any]], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    val_rows = sorted(
        [row for row in records if str(row.get("split", "")).startswith("val")],
        key=lambda row: (row.get("epoch", 0), row.get("global_step", 0)),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    f1_keys = sorted({key for row in val_rows for key in row if key.endswith("/scene_grasp/f1")})
    plotted = False
    for key in f1_keys:
        rows = [row for row in val_rows if key in row and row[key] not in ("", None)]
        if not rows:
            continue
        label = key.removesuffix("/scene_grasp/f1")
        ax.plot(
            [row["epoch"] for row in rows],
            [row[key] for row in rows],
            marker="o",
            linewidth=1.5,
            label=label,
        )
        plotted = True

    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No validation F1 series found.", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Validation Affordance F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)

    figure_path = output_path / "val_scene_grasp_f1.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def _latest_benchmark_rows(logdir: str | Path) -> list[dict[str, str]]:
    root = Path(logdir)
    if not root.exists():
        return []

    newest_by_scene: dict[str, Path] = {}
    for path in sorted(root.glob("*.json")):
        name = path.name
        if "_packed_" in name:
            newest_by_scene["packed"] = path
        elif "_pile_" in name:
            newest_by_scene["pile"] = path

    by_metric: list[dict[str, str]] = []
    for row in PAPER_BENCHMARK_ROWS:
        scene = row["scene"]
        metric = row["metric"]
        local_value = "-"
        if scene in newest_by_scene:
            with newest_by_scene[scene].open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if metric == "GSR":
                local_value = f"{as_percent_string(data.get('gsr'))} +/- {as_percent_string(data.get('gsr_std'))}"
            elif metric == "DR":
                local_value = f"{as_percent_string(data.get('dr'))} +/- {as_percent_string(data.get('dr_std'))}"
        by_metric.append(
            {
                "section": "benchmark",
                "paper_term": f"{scene} {metric}",
                "code_keys": newest_by_scene.get(scene, Path("-")).name if scene in newest_by_scene else "-",
                "paper_target": row["paper_value"],
                "notes": f"Local benchmark: {local_value}",
            }
        )
    return by_metric


def write_paper_comparison(records: list[dict[str, Any]], output_dir: str | Path, benchmark_logdir: str | Path) -> tuple[Path, Path]:
    del records
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = list(PAPER_TERM_ROWS) + _latest_benchmark_rows(benchmark_logdir)

    csv_path = output_path / "paper_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "paper_term", "code_keys", "paper_target", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_path / "paper_comparison.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# ICG-Net Training Diagnostics vs Paper\n\n")
        handle.write("## Training-Term Mapping\n\n")
        handle.write("| Paper Term | Code Keys | Paper Target | Notes |\n")
        handle.write("| --- | --- | --- | --- |\n")
        for row in PAPER_TERM_ROWS:
            handle.write(
                f"| {row['paper_term']} | {row['code_keys']} | {row['paper_target']} | {row['notes']} |\n"
            )

        handle.write("\n## Benchmark Comparison\n\n")
        handle.write("| Paper Term | Local Source | Paper Target | Notes |\n")
        handle.write("| --- | --- | --- | --- |\n")
        benchmark_rows = _latest_benchmark_rows(benchmark_logdir)
        if benchmark_rows:
            for row in benchmark_rows:
                handle.write(
                    f"| {row['paper_term']} | {row['code_keys']} | {row['paper_target']} | {row['notes']} |\n"
                )
        else:
            handle.write("| benchmark | - | 97.7 +/- 0.9 / 97.5 +/- 0.3 / 92.0 +/- 2.6 / 94.1 +/- 1.4 | No local benchmark logs found. |\n")

    return csv_path, md_path


def generate_training_report(metrics_source: str | Path, output_dir: str | Path | None = None, benchmark_logdir: str | Path = "logs/icg_full") -> dict[str, Path]:
    records = load_metric_records(metrics_source)
    destination = Path(output_dir) if output_dir is not None else metric_path_from_source(metrics_source).parent
    loss_plot = plot_training_losses(records, destination)
    f1_plot = plot_validation_f1(records, destination)
    csv_path, md_path = write_paper_comparison(records, destination, benchmark_logdir)
    return {
        "loss_plot": loss_plot,
        "f1_plot": f1_plot,
        "paper_csv": csv_path,
        "paper_md": md_path,
    }


class MetricsLogger:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        write_metric_records(self.records, self.run_dir)


class SyntheticSmokeDataset(Dataset):
    def __init__(
        self,
        length: int = 8,
        num_points: int = 48,
        num_instances: int = 2,
        num_sdf_points: int = 24,
        num_grasp_points: int = 16,
        num_classes: int = 7,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.length = length
        self.num_points = num_points
        self.num_instances = num_instances
        self.num_sdf_points = num_sdf_points
        self.num_grasp_points = num_grasp_points
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(self.seed + index)
        input_points = torch.rand((self.num_points, 3), generator=generator) * 0.2 + 0.05
        sdf_points = torch.rand((self.num_sdf_points, 3), generator=generator) * 0.2 + 0.05
        scene_grasp_points = torch.rand((self.num_grasp_points, 3), generator=generator) * 0.2 + 0.05

        centroids = torch.stack(
            [
                torch.tensor([0.10, 0.10, 0.10]),
                torch.tensor([0.20, 0.20, 0.12]),
            ]
        )[: self.num_instances]
        labels = torch.arange(self.num_instances, dtype=torch.long) % max(1, self.num_classes - 1)

        distances = torch.stack([(input_points - center).norm(dim=-1) for center in centroids], dim=0)
        assignments = distances.argmin(dim=0)
        masks = torch.stack([(assignments == idx).float() for idx in range(self.num_instances)], dim=0)

        sdf = []
        for center in centroids:
            sdf.append(0.04 - (sdf_points - center).norm(dim=-1))
        sdf = torch.stack(sdf, dim=0)

        scene_centric_labels = torch.zeros((self.num_instances, self.num_grasp_points, 13), dtype=torch.float32)
        for inst_idx, center in enumerate(centroids):
            affordance_seed = torch.rand((self.num_grasp_points, 12), generator=generator)
            affordance = (affordance_seed > 0.65).float()
            if affordance.sum() == 0:
                affordance[0, 0] = 1.0
            scene_centric_labels[inst_idx, :, :12] = affordance
            widths = (0.02 + 0.03 * torch.sigmoid(5 * (0.08 - (scene_grasp_points - center).norm(dim=-1)))).clamp(0, 0.08)
            widths[affordance.sum(dim=-1) == 0] = 0.0
            scene_centric_labels[inst_idx, :, 12] = widths

        target = {
            "labels": labels,
            "masks": masks,
            "sdf": sdf,
            "scene_centric_labels": scene_centric_labels,
        }
        return {
            "input_points": input_points,
            "scene_grasp_points": scene_grasp_points,
            "sdf_points": sdf_points,
            "targets": target,
            "mask_type": "masks",
        }


class SyntheticSmokeCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_points": [item["input_points"] for item in batch],
            "scene_grasp_points": [item["scene_grasp_points"] for item in batch],
            "sdf_points": [item["sdf_points"] for item in batch],
            "targets": [item["targets"] for item in batch],
            "mask_type": batch[0]["mask_type"],
        }


class SyntheticICGNetLikeModel(nn.Module):
    def __init__(self, num_queries: int = 4, num_classes: int = 7, hidden_dim: int = 32, **_: Any) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.grasp_decoder_type = "occnet_width"

        self.query_embeddings = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.1)
        self.query_positional = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.1)

        self.point_encoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.mask_query = nn.Linear(hidden_dim, hidden_dim)

        self.sdf_point_encoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.sdf_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        self.grasp_point_encoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.grasp_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 13))

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch_points = batch.get("input_points") or batch.get("raw_coordinates") or batch.get("coords")
        if batch_points is None:
            raise KeyError("SyntheticICGNetLikeModel expected 'input_points', 'raw_coordinates', or 'coords' in the batch.")
        batch_sdf_points = batch["sdf_points"]
        batch_grasp_points = batch["scene_grasp_points"]
        batch_size = len(batch_points)

        query_embeddings = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        query_positional = self.query_positional.unsqueeze(0).expand(batch_size, -1, -1)
        pred_logits = self.class_head(query_embeddings)

        pred_masks = []
        sdf_full = []
        scene_grasps_full = []
        for batch_idx, points in enumerate(batch_points):
            point_features = self.point_encoder(points)
            mask_queries = self.mask_query(query_embeddings[batch_idx])
            pred_masks.append(point_features @ mask_queries.transpose(0, 1))

            sdf_points = batch_sdf_points[batch_idx]
            sdf_features = self.sdf_point_encoder(sdf_points)
            sdf_joint = sdf_features.unsqueeze(0) + query_embeddings[batch_idx].unsqueeze(1)
            sdf_full.append(self.sdf_head(sdf_joint))

            grasp_points = batch_grasp_points[batch_idx]
            grasp_features = self.grasp_point_encoder(grasp_points)
            grasp_joint = grasp_features.unsqueeze(0) + query_embeddings[batch_idx].unsqueeze(1)
            scene_grasps_full.append(self.grasp_head(grasp_joint))

        return {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
            "instance_queries": query_embeddings,
            "all_queries": [query_embeddings],
            "instance_latents": query_embeddings + query_positional,
            "intermittent_latents": [query_embeddings + query_positional],
            "positional_encodings": query_positional,
            "scene_grasp_queries": [query_embeddings],
            "aux_outputs": [],
            "__sdf_pred_full": sdf_full,
            "__scene_grasps_full": scene_grasps_full,
        }


@dataclass
class ForwardResult:
    outputs: dict[str, Any]
    targets: list[dict[str, Any]]
    mask_type: str


class BaseBatchAdapter:
    def prepare_batch(self, batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        return move_to_device(batch, device)

    def forward_base(self, model: nn.Module, batch: dict[str, Any], device: torch.device) -> ForwardResult:
        raise NotImplementedError

    def attach_loss_predictions(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        outputs: dict[str, Any],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, Any]:
        del model, batch
        augmented = dict(outputs)
        if "__sdf_pred_full" in augmented:
            augmented["sdf_pred"] = [pred[src_idx] for pred, (src_idx, _) in zip(augmented.pop("__sdf_pred_full"), indices)]
        if "__scene_grasps_full" in augmented:
            augmented["scene_grasps"] = [
                pred[src_idx] for pred, (src_idx, _) in zip(augmented.pop("__scene_grasps_full"), indices)
            ]
        if "__object_grasps_full" in augmented:
            augmented["object_grasps"] = [
                pred[src_idx] for pred, (src_idx, _) in zip(augmented.pop("__object_grasps_full"), indices)
            ]
        return augmented

    def matched_scene_grasp_pairs(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if "scene_grasps" not in outputs:
            return []
        pairs = []
        for pred, target, (_, tgt_idx) in zip(outputs["scene_grasps"], targets, indices):
            if tgt_idx.numel() == 0:
                continue
            if "scene_centric_labels" not in target:
                continue
            pairs.append((target["scene_centric_labels"][tgt_idx], pred))
        return pairs


class SyntheticSmokeAdapter(BaseBatchAdapter):
    def forward_base(self, model: nn.Module, batch: dict[str, Any], device: torch.device) -> ForwardResult:
        del device
        outputs = model(batch)
        return ForwardResult(outputs=outputs, targets=batch["targets"], mask_type=batch.get("mask_type", "masks"))


class OfficialICGNetBatchAdapter(BaseBatchAdapter):
    def __init__(self, quantization_size: float = 0.003, require_cuda: bool = True) -> None:
        self.quantization_size = quantization_size
        self.require_cuda = require_cuda

    def _build_sparse_inputs(
        self,
        batch: dict[str, Any],
        device: torch.device,
    ) -> tuple[Any, list[torch.Tensor]]:
        try:
            import MinkowskiEngine as ME
            from torch_scatter import scatter_mean
        except ImportError as exc:
            raise RuntimeError("Official ICG-Net batches require MinkowskiEngine and torch_scatter in the active env.") from exc

        if "voxelized_data" in batch and "raw_coordinates" in batch:
            voxelized_data = batch["voxelized_data"]
            raw_coordinates = move_to_device(batch["raw_coordinates"], device)
            sparse_device = getattr(voxelized_data, "device", None)
            if sparse_device == device:
                return voxelized_data, raw_coordinates

            if hasattr(voxelized_data, "F") and hasattr(voxelized_data, "C"):
                rebuilt = ME.SparseTensor(
                    voxelized_data.F.to(device),
                    voxelized_data.C.to(device),
                    device=device,
                )
                return rebuilt, raw_coordinates

        coords = batch.get("coords")
        if coords is None:
            raise ValueError("Official adapter expected 'coords' or ('voxelized_data', 'raw_coordinates') in the batch.")

        coord_list = coords if isinstance(coords, list) else [coords]
        feature_list = batch.get("features")
        if feature_list is None:
            feature_list = [torch.ones((coord.shape[0], 1), dtype=coord.dtype, device=coord.device) for coord in coord_list]
        elif not isinstance(feature_list, list):
            feature_list = [feature_list]

        quantized_coords = []
        quantized_feats = []
        raw_coordinates = []
        for coord, feat in zip(coord_list, feature_list):
            coord = coord.to(device)
            feat = feat.to(device)
            qcoord, qfeat, _, inverse = ME.utils.sparse_quantize(
                coordinates=coord,
                features=feat,
                quantization_size=self.quantization_size,
                return_index=True,
                return_inverse=True,
            )
            quantized_coords.append(qcoord)
            quantized_feats.append(qfeat)
            raw_coordinates.append(scatter_mean(coord, inverse.to(device), dim=0))

        voxelized = ME.SparseTensor(
            torch.cat(quantized_feats, dim=0),
            ME.utils.batched_coordinates(quantized_coords),
            device=device,
        )
        return voxelized, raw_coordinates

    def forward_base(self, model: nn.Module, batch: dict[str, Any], device: torch.device) -> ForwardResult:
        if self.require_cuda and device.type != "cuda":
            raise RuntimeError(
                "The official ICG-Net forward path requires CUDA in this checkout because furthest-point sampling "
                "calls a GPU-only operator. Use a CUDA device for real training, or the synthetic smoke overlay for CPU smoke tests."
            )

        voxelized_data, raw_coordinates = self._build_sparse_inputs(batch, device)
        outputs = model(voxelized_data, raw_coordinates, poses=batch.get("poses"))
        outputs = dict(outputs)

        if "sdf_points" in batch:
            outputs["__sdf_pred_full"] = model.decode_sdf(
                list(outputs["instance_queries"]),
                list(outputs["positional_encodings"]),
                batch["sdf_points"],
            )
        if "scene_grasp_points" in batch:
            scene_queries = outputs["scene_grasp_queries"][-1]
            outputs["__scene_grasps_full"] = model.decode_grasps(
                list(scene_queries),
                list(outputs["positional_encodings"]),
                batch["scene_grasp_points"],
            )
        if "normal_points" in batch and "gt_normals" in batch and "gt_normals_id" in batch:
            outputs["normals"] = model.decode_normals(
                list(outputs["instance_queries"]),
                list(outputs["positional_encodings"]),
                batch["normal_points"],
            )
            outputs["gt_normals"] = batch["gt_normals"]
            outputs["gt_normals_id"] = batch["gt_normals_id"]

        return ForwardResult(outputs=outputs, targets=batch["targets"], mask_type=batch.get("mask_type", "masks"))


def instantiate_dataset(dataset_cfg: DictConfig | None) -> Dataset:
    if dataset_cfg is None:
        raise ValueError("Missing dataset config for the local train wrapper.")
    dataset = instantiate(dataset_cfg)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Instantiated dataset is not a torch Dataset: {type(dataset)}")
    return dataset


def instantiate_collator(collate_cfg: DictConfig | None) -> Any | None:
    if collate_cfg is None:
        return None
    return instantiate(collate_cfg)


def instantiate_dataloader(
    dataloader_cfg: DictConfig | None,
    dataset: Dataset,
    collate_fn: Any | None = None,
) -> DataLoader:
    if dataloader_cfg is None:
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    target = dataloader_cfg.get("_target_", "torch.utils.data.DataLoader")
    if target != "torch.utils.data.DataLoader":
        return instantiate(dataloader_cfg, dataset=dataset, collate_fn=collate_fn)

    kwargs = {
        key: value
        for key, value in omega_to_dict(dataloader_cfg).items()
        if key not in {"_target_", "dataset", "collate_fn"}
    }
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(dataset=dataset, **kwargs)


def create_run_dir(output_root: str | Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
