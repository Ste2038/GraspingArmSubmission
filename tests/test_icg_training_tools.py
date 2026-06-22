from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.icg_training_lib import SyntheticICGNetLikeModel, generate_training_report, write_metric_records
from scripts.train_icg import (
    capture_prefixed_state,
    capture_running_norm_state,
    configure_trainable_parameters,
    configured_accumulate_grad_batches,
    configured_validation_modes,
    restore_prefixed_state,
    restore_running_norm_state,
    set_validation_model_mode,
)
from scripts.sweep_icg_grasp_thresholds import compatible_model_state_dict, sweep_threshold_metrics, threshold_values


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "third_party/icg_benchmark/data/icgnet/51--0.656/config.yaml"
SMOKE_OVERLAY = REPO_ROOT / "configs/train_icg_smoke.yaml"


def test_generate_training_report_from_jsonl(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    records = [
        {
            "epoch": 1,
            "split": "train",
            "train/loss_inst_bce": 0.8,
            "train/loss_inst_dice": 0.7,
            "train/loss_sem_ce": 0.5,
            "train/loss_grasp_aff_scene": 0.6,
            "train/loss_grasp_width_scene": 0.2,
            "train/loss_occ_bce": 0.4,
        },
        {
            "epoch": 1,
            "split": "val",
            "val/scene_grasp/f1": 0.62,
        },
        {
            "epoch": 2,
            "split": "train",
            "train/loss_inst_bce": 0.6,
            "train/loss_inst_dice": 0.5,
            "train/loss_sem_ce": 0.4,
            "train/loss_grasp_aff_scene": 0.45,
            "train/loss_grasp_width_scene": 0.15,
            "train/loss_occ_bce": 0.3,
        },
        {
            "epoch": 2,
            "split": "val",
            "val/scene_grasp/f1": 0.68,
        },
    ]
    write_metric_records(records, metrics_dir)

    outputs = generate_training_report(metrics_dir)
    assert outputs["loss_plot"].exists()
    assert outputs["f1_plot"].exists()
    assert outputs["paper_csv"].exists()
    assert outputs["paper_md"].exists()

    markdown = outputs["paper_md"].read_text(encoding="utf-8")
    assert "Main validation metric" in markdown
    assert "loss_scene_occ is treated as an auxiliary scene-level occupancy term" in markdown


def test_generate_training_report_handles_missing_width_and_occ_main(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics_missing"
    records = [
        {
            "epoch": 1,
            "split": "train",
            "train/loss_inst_bce": 0.9,
            "train/loss_inst_dice": 0.8,
            "train/loss_sem_ce": 0.7,
            "train/loss_grasp_aff_scene": 0.6,
            "train/loss_occ_scene_aux": 0.4,
        },
        {
            "epoch": 1,
            "split": "val",
            "val/scene_grasp/f1": 0.55,
        },
    ]
    write_metric_records(records, metrics_dir)

    outputs = generate_training_report(metrics_dir)
    assert outputs["loss_plot"].exists()
    assert outputs["f1_plot"].exists()
    assert outputs["paper_csv"].exists()


def test_configured_validation_modes_accepts_norm_diagnostic_mode() -> None:
    assert configured_validation_modes({"validation_model_modes": "eval_train_norm"}) == ["eval_train_norm"]
    assert configured_validation_modes({"validation_model_modes": ["eval", "eval_train_norm", "train"]}) == [
        "eval",
        "eval_train_norm",
        "train",
    ]

    with pytest.raises(ValueError):
        configured_validation_modes({"validation_model_modes": ["eval", "bad_mode"]})


def test_configured_accumulate_grad_batches_uses_cli_then_wrapper() -> None:
    class Args:
        accumulate_grad_batches = None

    assert configured_accumulate_grad_batches(Args(), {}) == 1
    assert configured_accumulate_grad_batches(Args(), {"accumulate_grad_batches": 2}) == 2

    Args.accumulate_grad_batches = 3
    assert configured_accumulate_grad_batches(Args(), {"accumulate_grad_batches": 2}) == 3

    Args.accumulate_grad_batches = 0
    with pytest.raises(ValueError):
        configured_accumulate_grad_batches(Args(), {})


def test_configure_trainable_parameters_supports_prefix_ablation() -> None:
    model = SyntheticICGNetLikeModel(num_queries=4, num_classes=7, hidden_dim=16)

    report = configure_trainable_parameters(model, {"trainable_parameter_prefixes": ["grasp_head"]})

    assert report["matched_trainable_prefixes"]["grasp_head"] > 0
    assert report["trainable_tensors"] < report["total_tensors"]
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.startswith("grasp_head.")


def test_prefixed_state_capture_restores_reset_ablation() -> None:
    model = SyntheticICGNetLikeModel(num_queries=4, num_classes=7, hidden_dim=16)
    captured, report = capture_prefixed_state(model, ["grasp_head"])
    before = {name: value.clone() for name, value in captured.items()}
    untouched_before = model.state_dict()["point_encoder.0.weight"].clone()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    restore_prefixed_state(model, captured)
    state = model.state_dict()

    assert report["matched_reset_prefixes"]["grasp_head"] > 0
    assert report["reset_tensors"] == len(before)
    for name, value in before.items():
        assert torch.equal(state[name], value)
    assert torch.equal(state["point_encoder.0.weight"], untouched_before + 1.0)


def test_eval_train_norm_keeps_only_running_norm_layers_trainable() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.Dropout(p=0.5),
    )

    set_validation_model_mode(model, "eval_train_norm")

    assert not model.training
    assert not model[2].training
    assert model[1].training


def test_running_norm_state_can_be_restored_after_train_mode_validation() -> None:
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4))
    model.train()
    before = model[0].running_mean.detach().clone()

    state = capture_running_norm_state(model)
    model(torch.randn(8, 4))
    assert not torch.equal(model[0].running_mean, before)

    restore_running_norm_state(state)
    assert torch.equal(model[0].running_mean, before)


def test_threshold_sweep_reports_best_precision_recall_tradeoff() -> None:
    scores = torch.tensor([0.9, 0.8, 0.4, 0.1])
    labels = torch.tensor([1, 0, 1, 0]).bool()

    rows = sweep_threshold_metrics(scores, labels, threshold_values(0.25, 0.75, 0.25))

    assert [row["threshold"] for row in rows] == [0.25, 0.5, 0.75]
    assert rows[0]["recall"] == 1.0
    assert rows[1]["precision"] == 0.5
    assert rows[1]["recall"] == 0.5
    assert rows[2]["precision"] == 0.5
    assert rows[2]["recall"] == 0.5


def test_threshold_sweep_checkpoint_compatibility_strips_model_prefix() -> None:
    model = torch.nn.Linear(2, 1)
    source = {f"model.{key}": value.detach().clone() for key, value in model.state_dict().items()}

    compatible, report = compatible_model_state_dict(source, model)

    assert report["loaded"] == len(model.state_dict())
    assert report["missing_filled_from_init"] == 0
    assert set(compatible) == set(model.state_dict())


def test_train_wrapper_smoke_writes_metrics_and_supports_plotting(tmp_path: Path) -> None:
    output_root = tmp_path / "training_logs"
    init_model = SyntheticICGNetLikeModel(num_queries=4, num_classes=7, hidden_dim=32)
    init_checkpoint = tmp_path / "init_checkpoint.pt"
    torch.save(
        {"state_dict": {f"model.{key}": value.detach().clone() for key, value in init_model.state_dict().items()}},
        init_checkpoint,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/train_icg.py"),
            "--config",
            str(BASE_CONFIG),
            "--overlay",
            str(SMOKE_OVERLAY),
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--max-epochs",
            "1",
            "--run-name",
            "pytest_smoke",
            "--init-checkpoint",
            str(init_checkpoint),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    run_dirs = sorted(output_root.glob("pytest_smoke_*"))
    assert run_dirs, "Expected the smoke wrapper to create a run directory."
    run_dir = run_dirs[-1]
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "effective_config.yaml").exists()
    meta = json.loads((run_dir / "wrapper_meta.json").read_text(encoding="utf-8"))
    assert meta["init_checkpoint_report"]["loaded"] == len(init_model.state_dict())
    assert (run_dir / "checkpoint_last.pt").exists()
    assert (run_dir / "checkpoint_best.pt").exists()
    checkpoint = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["validation_record"]["split"] == "val"
    assert "state_dict" in checkpoint
    assert checkpoint["state_dict"].keys() == checkpoint["model_state_dict"].keys()

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
    assert (run_dir / "paper_comparison.md").exists()
