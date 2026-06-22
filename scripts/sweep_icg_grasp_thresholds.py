#!/usr/bin/env python3
"""Sweep grasp-affordance thresholds on reconstructed validation data."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.icg_dataset_lib import ensure_dir
from scripts.icg_training_lib import (
    compatible_model_state_dict,
    ensure_loss_weights,
    load_checkpoint_state_dict,
    merge_configs,
    scalarize,
    weighted_loss_sum,
)
from scripts.train_icg import (
    build_adapter,
    build_dataloaders,
    capture_running_norm_state,
    choose_device,
    set_seed,
    set_validation_model_mode,
    restore_running_norm_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-mode", choices=("eval", "eval_train_norm", "train"), default="eval_train_norm")
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    return parser.parse_args()


def threshold_values(start: float, end: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= end + (step / 10):
        values.append(round(current, 6))
        current += step
    return values


def flatten_scene_grasp_pair(labels: torch.Tensor, predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = predictions[..., :12].sigmoid().detach().float()
    gt = (labels[..., :12] > 0).detach()
    if pred.ndim == 3:
        pred = pred.max(dim=0)[0]
        gt = gt.any(dim=0)
    return pred.reshape(-1).cpu(), gt.reshape(-1).cpu()


def sweep_threshold_metrics(scores: torch.Tensor, labels: torch.Tensor, thresholds: list[float]) -> list[dict[str, float]]:
    label_bool = labels.bool()
    total = int(label_bool.numel())
    positives = int(label_bool.sum().item())
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred_bool = scores >= threshold
        tp = int((pred_bool & label_bool).sum().item())
        fp = int((pred_bool & ~label_bool).sum().item())
        fn = int((~pred_bool & label_bool).sum().item())
        tn = int((~pred_bool & ~label_bool).sum().item())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": threshold,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": (tp + tn) / total if total else 0.0,
                "predicted_positive_fraction": (tp + fp) / total if total else 0.0,
                "target_positive_fraction": positives / total if total else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return rows


def score_summary(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    label_bool = labels.bool()
    positive_scores = scores[label_bool]
    negative_scores = scores[~label_bool]
    return {
        "score_mean": float(scores.mean().item()) if scores.numel() else 0.0,
        "score_min": float(scores.min().item()) if scores.numel() else 0.0,
        "score_max": float(scores.max().item()) if scores.numel() else 0.0,
        "positive_score_mean": float(positive_scores.mean().item()) if positive_scores.numel() else 0.0,
        "negative_score_mean": float(negative_scores.mean().item()) if negative_scores.numel() else 0.0,
        "positive_count": int(label_bool.sum().item()),
        "total_count": int(label_bool.numel()),
    }


def write_outputs(
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    threshold_rows: list[dict[str, float]],
    batch_rows: list[dict[str, Any]],
    score_stats: dict[str, float],
) -> dict[str, Path]:
    output_path = ensure_dir(output_dir)
    json_path = output_path / "threshold_sweep.json"
    csv_path = output_path / "threshold_sweep.csv"
    batch_csv_path = output_path / "threshold_sweep_batches.csv"
    md_path = output_path / "threshold_sweep.md"

    best_row = max(threshold_rows, key=lambda row: row["f1"])
    payload = {
        "metadata": metadata,
        "best_threshold": best_row,
        "score_summary": score_stats,
        "thresholds": threshold_rows,
        "batches": batch_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_rows)

    with batch_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in batch_rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(batch_rows)

    lines = [
        "# ICG-Net Reconstructed Grasp Threshold Sweep",
        "",
        f"- Generated: `{metadata['generated_at']}`",
        f"- Checkpoint: `{metadata['checkpoint']}`",
        f"- Validation mode: `{metadata['validation_mode']}`",
        f"- Batches: `{metadata['batches']}`",
        f"- Matched label slots: `{score_stats['total_count']}`",
        f"- Target positive fraction: `{best_row['target_positive_fraction']:.6f}`",
        "",
        "## Best Threshold",
        "",
        "| Threshold | F1 | Precision | Recall | Predicted Positive Fraction |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| {threshold:.2f} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {predicted_positive_fraction:.4f} |".format(
            **best_row
        ),
        "",
        "## Score Summary",
        "",
        f"- Mean score, positives: `{score_stats['positive_score_mean']:.6f}`",
        f"- Mean score, negatives: `{score_stats['negative_score_mean']:.6f}`",
        f"- Overall score range: `{score_stats['score_min']:.6f} .. {score_stats['score_max']:.6f}`",
        "",
        "## Full Sweep",
        "",
        "| Threshold | F1 | Precision | Recall | Accuracy | Predicted Positive Fraction |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in threshold_rows:
        lines.append(
            "| {threshold:.2f} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {accuracy:.4f} | {predicted_positive_fraction:.4f} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "batch_csv": batch_csv_path, "markdown": md_path}


def run_sweep(args: argparse.Namespace) -> dict[str, Path]:
    cfg = merge_configs(args.config, args.overlay)
    set_seed(args.seed)
    device = choose_device(args.device)
    loss_weights = ensure_loss_weights(cfg)
    _, val_loader = build_dataloaders(cfg)
    adapter = build_adapter(cfg)
    model = instantiate(cfg.model).to(device)
    state_dict = load_checkpoint_state_dict(args.checkpoint, device)
    state_dict, load_report = compatible_model_state_dict(state_dict, model)
    model.load_state_dict(state_dict)
    criterion = instantiate(cfg.loss, matcher=instantiate(cfg.matcher), weight_dict=loss_weights).to(device)
    criterion.eval()

    norm_state = capture_running_norm_state(model) if args.validation_mode in {"train", "eval_train_norm"} else []
    set_validation_model_mode(model, args.validation_mode)

    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    batch_rows: list[dict[str, Any]] = []
    total_losses = []

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, start=1):
                if args.max_val_batches is not None and batch_idx > args.max_val_batches:
                    break
                prepared = adapter.prepare_batch(batch, device)
                forward_result = adapter.forward_base(model, prepared, device)
                indices = criterion.get_matches(
                    forward_result.outputs,
                    forward_result.targets,
                    forward_result.mask_type,
                )
                loss_outputs = adapter.attach_loss_predictions(model, prepared, forward_result.outputs, indices)
                losses = criterion(
                    loss_outputs,
                    forward_result.targets,
                    forward_result.mask_type,
                    indices=indices,
                )
                total_loss, _ = weighted_loss_sum(losses, loss_weights)
                total_losses.append(scalarize(total_loss))
                pairs = adapter.matched_scene_grasp_pairs(loss_outputs, forward_result.targets, indices)
                for pair_idx, (labels, predictions) in enumerate(pairs):
                    scores, target = flatten_scene_grasp_pair(labels, predictions)
                    all_scores.append(scores)
                    all_labels.append(target)
                    meta = prepared.get("meta", [{}])
                    batch_meta = meta[0] if isinstance(meta, list) and meta else {}
                    batch_rows.append(
                        {
                            "batch": batch_idx,
                            "pair": pair_idx,
                            "scene_id": batch_meta.get("scene_id", ""),
                            "loss_total_weighted": scalarize(total_loss),
                            "score_mean": float(scores.mean().item()) if scores.numel() else 0.0,
                            "target_positive_fraction": float(target.float().mean().item()) if target.numel() else 0.0,
                            "slots": int(target.numel()),
                        }
                    )
    finally:
        restore_running_norm_state(norm_state)

    if not all_scores:
        raise ValueError("No matched scene-grasp predictions were collected.")

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    thresholds = threshold_values(args.threshold_start, args.threshold_end, args.threshold_step)
    threshold_rows = sweep_threshold_metrics(scores, labels, thresholds)
    stats = score_summary(scores, labels)
    stats["mean_loss_total_weighted"] = sum(total_losses) / len(total_losses) if total_losses else 0.0

    default_output_dir = args.checkpoint.resolve().parent / "threshold_sweep" / args.validation_mode
    output_dir = (args.output_dir or default_output_dir).resolve()
    metadata = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": str(args.config.resolve()),
        "overlays": [str(path.resolve()) for path in args.overlay],
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "validation_mode": args.validation_mode,
        "batches": len(batch_rows),
        "mean_loss_total_weighted": stats["mean_loss_total_weighted"],
        "checkpoint_load_report": load_report,
    }
    return write_outputs(
        output_dir,
        metadata=metadata,
        threshold_rows=threshold_rows,
        batch_rows=batch_rows,
        score_stats=stats,
    )


def main() -> None:
    args = parse_args()
    outputs = run_sweep(args)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
