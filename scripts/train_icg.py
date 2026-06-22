#!/usr/bin/env python3
"""Local ICG-Net training wrapper with explicit epoch metrics and paper-oriented aliases."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
import time

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.icg_training_lib import (
    MetricsLogger,
    OfficialICGNetBatchAdapter,
    SyntheticSmokeAdapter,
    average_metric_dicts,
    build_alias_metrics,
    create_run_dir,
    ensure_loss_weights,
    initialize_model_from_checkpoint,
    instantiate_collator,
    instantiate_dataloader,
    instantiate_dataset,
    load_yaml_config,
    merge_configs,
    scalarize,
    write_metric_records,
    weighted_loss_sum,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base ICG-Net YAML config (for example the official checkpoint config).",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        action="append",
        default=[],
        help="Overlay YAML with wrapper-specific settings, explicit loss weights, and optional dataset/model overrides.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("logs/training"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional model checkpoint used only to initialize weights before training.",
    )
    parser.add_argument(
        "--log-batches",
        action="store_true",
        help="Print per-batch timing and lightweight progress information.",
    )
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    wrapper = cfg.get("wrapper", {})
    data_cfg = cfg.get("data", {})

    train_dataset_cfg = wrapper.get("train_dataset", data_cfg.get("train_dataset"))
    val_dataset_cfg = wrapper.get("validation_dataset", data_cfg.get("validation_dataset"))
    train_collate_cfg = wrapper.get("train_collate", data_cfg.get("train_collation"))
    val_collate_cfg = wrapper.get("validation_collate", data_cfg.get("validation_collation"))
    train_loader_cfg = wrapper.get("train_dataloader", data_cfg.get("train_dataloader"))
    val_loader_cfg = wrapper.get("validation_dataloader", data_cfg.get("validation_dataloader"))

    train_dataset = instantiate_dataset(train_dataset_cfg)
    val_dataset = instantiate_dataset(val_dataset_cfg)
    train_collate = instantiate_collator(train_collate_cfg)
    val_collate = instantiate_collator(val_collate_cfg)
    train_loader = instantiate_dataloader(train_loader_cfg, train_dataset, train_collate)
    val_loader = instantiate_dataloader(val_loader_cfg, val_dataset, val_collate)
    return train_loader, val_loader


def build_adapter(cfg):
    adapter_cfg = cfg.get("wrapper", {}).get("adapter")
    if adapter_cfg is None:
        quantization_size = float(cfg.get("data", {}).get("voxel_size", 0.003))
        return OfficialICGNetBatchAdapter(quantization_size=quantization_size)
    return instantiate(adapter_cfg)


def build_optimizer_and_scheduler(cfg, model: torch.nn.Module):
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable model parameters remain after wrapper freeze/trainable configuration.")
    optimizer = instantiate(cfg.optimizer, params=trainable_parameters)
    scheduler = None
    scheduler_interval = None
    scheduler_cfg = cfg.get("scheduler")
    if scheduler_cfg is not None and scheduler_cfg.get("scheduler") is not None:
        scheduler = instantiate(scheduler_cfg.scheduler, optimizer=optimizer)
        scheduler_interval = scheduler_cfg.get("pytorch_lightning_params", {}).get("interval", "epoch")
    return optimizer, scheduler, scheduler_interval


def as_float_dict(losses: dict[str, torch.Tensor], total_loss: torch.Tensor) -> dict[str, float]:
    metrics = {key: scalarize(value) for key, value in losses.items()}
    metrics["loss_total_weighted"] = scalarize(total_loss)
    return metrics


def update_grasp_metrics(grasp_metrics, adapter, outputs, targets, indices, decoder_type: str) -> int:
    updates = 0
    for labels, preds in adapter.matched_scene_grasp_pairs(outputs, targets, indices):
        grasp_metrics(labels.detach(), preds.detach(), decoder_type=decoder_type)
        updates += 1
    return updates


def epoch_record(
    epoch: int,
    split: str,
    loss_metrics: dict[str, float],
    sdf_loss_type: str | None,
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, float | int | str]:
    record: dict[str, float | int | str] = {"epoch": epoch, "split": split}
    record.update(loss_metrics)
    record.update(build_alias_metrics(split, loss_metrics, sdf_loss_type))
    if extra_metrics:
        record.update(extra_metrics)
    return record


VALIDATION_MODEL_MODES = {"eval", "train", "eval_train_norm"}


def configured_validation_modes(wrapper_cfg) -> list[str]:
    raw_modes = wrapper_cfg.get("validation_model_modes")
    if raw_modes is None:
        raw_modes = [wrapper_cfg.get("validation_model_mode", "eval")]
    elif isinstance(raw_modes, str):
        raw_modes = [raw_modes]

    modes = [str(mode) for mode in raw_modes]
    invalid = [mode for mode in modes if mode not in VALIDATION_MODEL_MODES]
    if invalid:
        raise ValueError(f"wrapper validation modes must be one of {sorted(VALIDATION_MODEL_MODES)}.")
    if not modes:
        raise ValueError("At least one wrapper validation mode is required.")
    return modes


def is_running_norm_module(module: torch.nn.Module) -> bool:
    name = type(module).__name__
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return True
    return "BatchNorm" in name or "InstanceNorm" in name


def set_validation_model_mode(model: torch.nn.Module, mode: str) -> None:
    if mode == "train":
        model.train()
        return
    model.eval()
    if mode == "eval_train_norm":
        for module in model.modules():
            if is_running_norm_module(module):
                module.train()


def capture_running_norm_state(model: torch.nn.Module) -> list[tuple[torch.nn.Module, dict[str, torch.Tensor]]]:
    states: list[tuple[torch.nn.Module, dict[str, torch.Tensor]]] = []
    for module in model.modules():
        if not is_running_norm_module(module):
            continue
        buffers = {
            name: value.detach().clone()
            for name, value in module.named_buffers(recurse=False)
            if value is not None
        }
        if buffers:
            states.append((module, buffers))
    return states


def restore_running_norm_state(states: list[tuple[torch.nn.Module, dict[str, torch.Tensor]]]) -> None:
    with torch.no_grad():
        for module, buffers in states:
            for name, value in buffers.items():
                current = getattr(module, name, None)
                if current is not None:
                    current.copy_(value.to(device=current.device, dtype=current.dtype))


def default_checkpoint_split(validation_model_modes: list[str]) -> str:
    if len(validation_model_modes) == 1:
        return "val"
    if "eval_train_norm" in validation_model_modes:
        return "val_eval_train_norm"
    return f"val_{validation_model_modes[0]}"


def checkpoint_score(record: dict[str, float | int | str], metric: str) -> float:
    if metric not in record or record[metric] == "":
        raise ValueError(f"Checkpoint metric {metric!r} is missing from split {record.get('split')!r}.")
    return float(record[metric])


def is_better_checkpoint(score: float, best_score: float | None, mode: str) -> bool:
    if best_score is None:
        return True
    if mode == "min":
        return score < best_score
    if mode == "max":
        return score > best_score
    raise ValueError("wrapper.checkpoint_mode must be 'min' or 'max'.")


def configured_accumulate_grad_batches(args: argparse.Namespace, wrapper_cfg) -> int:
    value = args.accumulate_grad_batches
    if value is None:
        value = wrapper_cfg.get("accumulate_grad_batches", 1)
    value = int(value)
    if value < 1:
        raise ValueError("accumulate_grad_batches must be >= 1.")
    return value


def normalize_prefix_list(raw_value) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [raw_value]
    return [str(value) for value in raw_value]


def parameter_matches_prefix(name: str, prefixes: list[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def configure_trainable_parameters(model: torch.nn.Module, wrapper_cfg) -> dict[str, int | list[str] | dict[str, int]]:
    trainable_prefixes = normalize_prefix_list(wrapper_cfg.get("trainable_parameter_prefixes"))
    freeze_prefixes = normalize_prefix_list(wrapper_cfg.get("freeze_parameter_prefixes"))

    trainable_matches = {prefix: 0 for prefix in trainable_prefixes}
    freeze_matches = {prefix: 0 for prefix in freeze_prefixes}

    if trainable_prefixes:
        for _, parameter in model.named_parameters():
            parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        for prefix in trainable_prefixes:
            if parameter_matches_prefix(name, [prefix]):
                parameter.requires_grad = True
                trainable_matches[prefix] += 1
        for prefix in freeze_prefixes:
            if parameter_matches_prefix(name, [prefix]):
                parameter.requires_grad = False
                freeze_matches[prefix] += 1

    if trainable_prefixes:
        unmatched = [prefix for prefix, count in trainable_matches.items() if count == 0]
        if unmatched:
            raise ValueError(f"wrapper.trainable_parameter_prefixes did not match any parameters: {unmatched}")
    if freeze_prefixes:
        unmatched = [prefix for prefix, count in freeze_matches.items() if count == 0]
        if unmatched:
            raise ValueError(f"wrapper.freeze_parameter_prefixes did not match any parameters: {unmatched}")

    total_tensors = 0
    trainable_tensors = 0
    total_parameters = 0
    trainable_parameters = 0
    for _, parameter in model.named_parameters():
        total_tensors += 1
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_tensors += 1
            trainable_parameters += parameter.numel()

    return {
        "trainable_parameter_prefixes": trainable_prefixes,
        "freeze_parameter_prefixes": freeze_prefixes,
        "matched_trainable_prefixes": trainable_matches,
        "matched_freeze_prefixes": freeze_matches,
        "total_tensors": total_tensors,
        "trainable_tensors": trainable_tensors,
        "frozen_tensors": total_tensors - trainable_tensors,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": total_parameters - trainable_parameters,
    }


def capture_prefixed_state(
    model: torch.nn.Module,
    prefixes: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, int | list[str] | dict[str, int]]]:
    matches = {prefix: 0 for prefix in prefixes}
    captured: dict[str, torch.Tensor] = {}
    if not prefixes:
        return captured, {
            "reset_parameter_prefixes": [],
            "matched_reset_prefixes": matches,
            "reset_tensors": 0,
            "reset_scalars": 0,
        }

    for name, value in model.state_dict().items():
        for prefix in prefixes:
            if parameter_matches_prefix(name, [prefix]):
                captured[name] = value.detach().clone()
                matches[prefix] += 1
                break

    unmatched = [prefix for prefix, count in matches.items() if count == 0]
    if unmatched:
        raise ValueError(f"wrapper.reset_parameter_prefixes did not match any state entries: {unmatched}")

    return captured, {
        "reset_parameter_prefixes": prefixes,
        "matched_reset_prefixes": matches,
        "reset_tensors": len(captured),
        "reset_scalars": sum(value.numel() for value in captured.values()),
    }


def restore_prefixed_state(model: torch.nn.Module, captured: dict[str, torch.Tensor]) -> None:
    if not captured:
        return
    model_state = model.state_dict()
    with torch.no_grad():
        for name, value in captured.items():
            if name not in model_state:
                raise ValueError(f"Cannot reset missing model state entry {name!r}.")
            model_state[name].copy_(value.to(device=model_state[name].device, dtype=model_state[name].dtype))


def save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg,
    train_record: dict[str, float | int | str],
    validation_record: dict[str, float | int | str],
    checkpoint_metric: str,
    checkpoint_score_value: float,
) -> None:
    model_state_dict = model.state_dict()
    payload = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_record": train_record,
        "validation_record": validation_record,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_score": checkpoint_score_value,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    cfg = merge_configs(args.config, args.overlay)
    device = choose_device(args.device)

    default_seed = 0
    if cfg.get("general") is not None and cfg.general.get("seed") is not None:
        default_seed = int(cfg.general.seed)
    seed = int(args.seed) if args.seed is not None else default_seed
    set_seed(seed)

    wrapper_cfg = cfg.get("wrapper", {})
    loss_weights = ensure_loss_weights(cfg)
    max_epochs = int(args.max_epochs or wrapper_cfg.get("max_epochs") or cfg.get("trainer", {}).get("max_epochs", 1))
    max_train_batches = args.max_train_batches or wrapper_cfg.get("max_train_batches")
    max_val_batches = args.max_val_batches or wrapper_cfg.get("max_val_batches")
    log_batches = bool(args.log_batches or wrapper_cfg.get("log_batches", False))
    validation_model_modes = configured_validation_modes(wrapper_cfg)
    save_checkpoints = bool(wrapper_cfg.get("save_checkpoints", True))
    checkpoint_split = str(wrapper_cfg.get("checkpoint_split") or default_checkpoint_split(validation_model_modes))
    checkpoint_metric = str(wrapper_cfg.get("checkpoint_metric", "loss_total_weighted"))
    checkpoint_mode = str(wrapper_cfg.get("checkpoint_mode", "min")).lower()
    accumulate_grad_batches = configured_accumulate_grad_batches(args, wrapper_cfg)
    init_checkpoint = args.init_checkpoint or wrapper_cfg.get("init_checkpoint")
    best_checkpoint_score: float | None = None
    run_name = args.run_name or wrapper_cfg.get("run_name") or cfg.get("general", {}).get("experiment_name") or "icg_train"
    run_dir = create_run_dir(args.output_root, run_name)
    metrics_logger = MetricsLogger(run_dir)

    (run_dir / "effective_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    (run_dir / "wrapper_meta.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "seed": seed,
                "loss_weights": loss_weights,
                "validation_model_modes": validation_model_modes,
                "save_checkpoints": save_checkpoints,
                "checkpoint_split": checkpoint_split,
                "checkpoint_metric": checkpoint_metric,
                "checkpoint_mode": checkpoint_mode,
                "accumulate_grad_batches": accumulate_grad_batches,
                "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    train_loader, val_loader = build_dataloaders(cfg)
    adapter = build_adapter(cfg)
    model = instantiate(cfg.model).to(device)
    reset_prefixes = normalize_prefix_list(wrapper_cfg.get("reset_parameter_prefixes"))
    reset_state, reset_report = capture_prefixed_state(model, reset_prefixes)
    init_checkpoint_report = None
    if init_checkpoint:
        init_checkpoint_report = initialize_model_from_checkpoint(model, init_checkpoint, device)
        meta_path = run_dir / "wrapper_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["init_checkpoint_report"] = init_checkpoint_report
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "Initialized model from "
            f"{init_checkpoint_report['checkpoint']} "
            f"(loaded={init_checkpoint_report['loaded']}, "
            f"missing={init_checkpoint_report['missing_filled_from_init']}, "
            f"ignored={init_checkpoint_report['excessive_ignored']})",
            flush=True,
        )
    if reset_prefixes:
        restore_prefixed_state(model, reset_state)
        meta_path = run_dir / "wrapper_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["reset_parameter_report"] = reset_report
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "Reset initialized state entries "
            f"(tensors={reset_report['reset_tensors']}, scalars={reset_report['reset_scalars']})",
            flush=True,
        )
    trainable_report = configure_trainable_parameters(model, wrapper_cfg)
    meta_path = run_dir / "wrapper_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["trainable_parameter_report"] = trainable_report
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    if trainable_report["frozen_tensors"]:
        print(
            "Configured trainable parameters "
            f"(trainable={trainable_report['trainable_parameters']}/"
            f"{trainable_report['total_parameters']} scalars, "
            f"trainable_tensors={trainable_report['trainable_tensors']}/"
            f"{trainable_report['total_tensors']})",
            flush=True,
        )
    criterion = instantiate(cfg.loss, matcher=instantiate(cfg.matcher), weight_dict=loss_weights)
    criterion = criterion.to(device)
    optimizer, scheduler, scheduler_interval = build_optimizer_and_scheduler(cfg, model)

    grasp_decoder_type = getattr(model, "grasp_decoder_type", cfg.get("model", {}).get("grasp_decoder", "occnet_width"))
    sdf_loss_type = cfg.get("loss", {}).get("sdf_loss_type")

    from icg_net.trainer.eval.metric import GraspMetrics

    for epoch in range(1, max_epochs + 1):
        if log_batches:
            print(f"[epoch {epoch}/{max_epochs}] train start", flush=True)
        model.train()
        criterion.train()
        train_batch_metrics: list[dict[str, float]] = []
        optimizer.zero_grad(set_to_none=True)
        pending_accumulation_steps = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            if max_train_batches is not None and batch_idx > int(max_train_batches):
                break
            batch_start = time.perf_counter()
            prepared = adapter.prepare_batch(batch, device)
            forward_result = adapter.forward_base(model, prepared, device)
            indices = criterion.get_matches(forward_result.outputs, forward_result.targets, forward_result.mask_type)
            loss_outputs = adapter.attach_loss_predictions(model, prepared, forward_result.outputs, indices)
            losses = criterion(
                loss_outputs,
                forward_result.targets,
                forward_result.mask_type,
                indices=indices,
            )
            total_loss, _ = weighted_loss_sum(losses, loss_weights)

            (total_loss / accumulate_grad_batches).backward()
            pending_accumulation_steps += 1
            if pending_accumulation_steps >= accumulate_grad_batches:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_accumulation_steps = 0
                if scheduler is not None and scheduler_interval == "step":
                    scheduler.step()

            train_batch_metrics.append(as_float_dict(losses, total_loss))
            if log_batches:
                elapsed = time.perf_counter() - batch_start
                print(
                    f"[epoch {epoch}/{max_epochs}] train batch {batch_idx} "
                    f"loss={scalarize(total_loss):.4f} time={elapsed:.2f}s",
                    flush=True,
                )

        if pending_accumulation_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and scheduler_interval == "step":
                scheduler.step()

        if scheduler is not None and scheduler_interval != "step":
            scheduler.step()

        train_loss_metrics = average_metric_dicts(train_batch_metrics)
        train_record = epoch_record(epoch, "train", train_loss_metrics, sdf_loss_type)
        metrics_logger.append(train_record)
        validation_records: list[dict[str, float | int | str]] = []

        for validation_model_mode in validation_model_modes:
            split_name = (
                "val"
                if len(validation_model_modes) == 1
                else f"val_{validation_model_mode}"
            )
            if log_batches:
                print(f"[epoch {epoch}/{max_epochs}] {split_name} start", flush=True)
            norm_state = (
                capture_running_norm_state(model)
                if validation_model_mode in {"train", "eval_train_norm"}
                else []
            )
            set_validation_model_mode(model, validation_model_mode)
            criterion.eval()
            val_batch_metrics: list[dict[str, float]] = []
            grasp_metrics = GraspMetrics().to(device)
            grasp_metric_updates = 0

            try:
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader, start=1):
                        if max_val_batches is not None and batch_idx > int(max_val_batches):
                            break
                        batch_start = time.perf_counter()
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
                        val_batch_metrics.append(as_float_dict(losses, total_loss))
                        grasp_metric_updates += update_grasp_metrics(
                            grasp_metrics,
                            adapter,
                            loss_outputs,
                            forward_result.targets,
                            indices,
                            grasp_decoder_type,
                        )
                        if log_batches:
                            elapsed = time.perf_counter() - batch_start
                            print(
                                f"[epoch {epoch}/{max_epochs}] {split_name} batch {batch_idx} "
                                f"loss={scalarize(total_loss):.4f} time={elapsed:.2f}s",
                                flush=True,
                            )
            finally:
                restore_running_norm_state(norm_state)

            val_loss_metrics = average_metric_dicts(val_batch_metrics)
            val_extra_metrics = {}
            if grasp_metric_updates > 0:
                val_extra_metrics = {
                    key: scalarize(value)
                    for key, value in grasp_metrics.get_metrics(prefix=f"{split_name}/scene_grasp").items()
                }
            val_record = epoch_record(epoch, split_name, val_loss_metrics, sdf_loss_type, val_extra_metrics)
            metrics_logger.append(val_record)
            validation_records.append(val_record)

        if save_checkpoints:
            selected_record = next((record for record in validation_records if record["split"] == checkpoint_split), None)
            if selected_record is None:
                available = ", ".join(str(record["split"]) for record in validation_records)
                raise ValueError(f"Checkpoint split {checkpoint_split!r} not found. Available validation splits: {available}")
            score = checkpoint_score(selected_record, checkpoint_metric)
            save_training_checkpoint(
                run_dir / "checkpoint_last.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                train_record=train_record,
                validation_record=selected_record,
                checkpoint_metric=checkpoint_metric,
                checkpoint_score_value=score,
            )
            if is_better_checkpoint(score, best_checkpoint_score, checkpoint_mode):
                best_checkpoint_score = score
                save_training_checkpoint(
                    run_dir / "checkpoint_best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    train_record=train_record,
                    validation_record=selected_record,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_score_value=score,
                )
                (run_dir / "checkpoint_best.json").write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "split": selected_record["split"],
                            "metric": checkpoint_metric,
                            "mode": checkpoint_mode,
                            "score": score,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )

    write_metric_records(metrics_logger.records, run_dir)
    print(f"Training metrics written to {run_dir}")


if __name__ == "__main__":
    main()
