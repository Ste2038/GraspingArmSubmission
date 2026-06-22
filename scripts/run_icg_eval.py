#!/usr/bin/env python3
"""Run the official ICG-Net benchmark with explicit reproducibility options."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ICG_NET_DIR = REPO_ROOT / "third_party" / "icg_net"
ICG_BENCHMARK_DIR = REPO_ROOT / "third_party" / "icg_benchmark"

sys.path.insert(0, str(ICG_NET_DIR))
sys.path.insert(0, str(ICG_BENCHMARK_DIR))

try:
    from icg_benchmark.grasping.eval import GraspEvaluator
    from icg_benchmark.grasping.planners import ICGNetPlanner
    from icg_benchmark.grasping.preprocessing import ICGNetObservation
    from icg_benchmark.grasping.view_samplers.top_down_view import sample_top_down_view
    from icg_net import ICGNetModule
except ImportError as exc:
    raise SystemExit(
        "Failed to import ICG-Net dependencies. Run setup_icg_env.sh and activate the icg_cuda121 environment."
    ) from exc


def default_config() -> Path:
    expected = ICG_BENCHMARK_DIR / "data" / "icgnet" / "51--0.656" / "config.yaml"
    if expected.exists():
        return expected

    legacy = ICG_BENCHMARK_DIR / "data" / "51--0.656" / "config.yaml"
    if legacy.exists():
        return legacy

    candidates = sorted((ICG_BENCHMARK_DIR / "data").glob("**/config.yaml"))
    if candidates:
        return candidates[0]

    raise SystemExit("No ICG-Net config.yaml found. Run scripts/download_icg_data.sh first.")


def config_with_local_checkpoint(config_path: Path, logdir: Path, checkpoint_override: Path | None = None) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle)

    general = config_data.setdefault("general", {})
    if checkpoint_override is not None:
        if not checkpoint_override.exists():
            raise SystemExit(f"Checkpoint override does not exist: {checkpoint_override}")
        output_dir = logdir / "resolved_configs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{config_path.parent.name}_{checkpoint_override.stem}_config.yaml"
        general["checkpoint"] = os.fspath(checkpoint_override.resolve())
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config_data, handle, sort_keys=False)
        print(f"[icg] Using checkpoint override: {general['checkpoint']}")
        return output_path

    checkpoint = Path(general.get("checkpoint") or "")
    if checkpoint.exists():
        return config_path

    candidates = [
        config_path.parent / "checkpoint.ckpt",
        ICG_BENCHMARK_DIR / "data" / "icgnet" / config_path.parent.name / "checkpoint.ckpt",
        ICG_BENCHMARK_DIR / "data" / config_path.parent.name / "checkpoint.ckpt",
    ]
    for candidate in candidates:
        if candidate.exists():
            output_dir = logdir / "resolved_configs"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{config_path.parent.name}_config.yaml"
            general["checkpoint"] = os.fspath(candidate.resolve())
            with output_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config_data, handle, sort_keys=False)
            print(f"[icg] Rewrote checkpoint path for local run: {general['checkpoint']}")
            return output_path

    raise SystemExit(
        f"Checkpoint from config does not exist ({checkpoint}) and no local checkpoint.ckpt was found next to "
        f"{config_path}."
    )


def build_model(args: argparse.Namespace) -> ICGNetModule:
    return ICGNetModule(
        config=str(args.config),
        device=args.device,
        grasp_each_object=True,
        n_grasps=args.n_grasps,
        n_grasp_pred_orientations=args.n_ori,
        gripper_offset=0.0,
        gripper_offset_perc=10.5,
        max_gripper_width=args.max_gripper_width,
        full_width=args.full_width,
        coll_checks=args.coll_checks,
    ).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["packed", "pile", "obj", "egad"], default="packed")
    parser.add_argument("--object-set", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--logdir", type=Path, default=REPO_ROOT / "logs" / "icg")
    parser.add_argument("--name", default="icgnet")
    parser.add_argument("--num-runs", type=int, default=4)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--num-views", type=int, default=1)
    parser.add_argument("--object-count", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--n-grasps", type=int, default=8192)
    parser.add_argument("--n-ori", type=int, default=6)
    parser.add_argument("--max-gripper-width", type=float, default=0.08)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--top-down", action="store_true")
    parser.add_argument("--with-table", action="store_true")
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--latent-replay", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--fps", action="store_true")
    parser.add_argument("--rand", dest="rand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-width", dest="full_width", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coll-checks", dest="coll_checks", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.config is None:
        args.config = default_config()
    else:
        args.config = args.config.resolve()
    if args.object_set is None:
        args.object_set = f"{args.scene}/test"
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()

    args.logdir = args.logdir.resolve()
    args.logdir.mkdir(parents=True, exist_ok=True)
    args.config = config_with_local_checkpoint(args.config, args.logdir, args.checkpoint)

    # icg_benchmark uses relative paths such as ./data/urdfs internally.
    # Keep wrapper inputs absolute, then run the evaluator from its repo root.
    os.chdir(ICG_BENCHMARK_DIR)

    def preproc(**kwargs):
        return ICGNetObservation(with_table=args.with_table, **kwargs)

    evaluator = GraspEvaluator(
        scene=args.scene,
        object_set=args.object_set,
        show_gui=args.sim_gui,
        rand=args.rand,
        preproc=preproc,
    )
    model = build_model(args)
    planner = ICGNetPlanner(
        model,
        device=args.device,
        confidence_th=args.threshold,
        resample=args.resample,
        visualize=args.vis,
        latent_imagination=args.latent_replay,
        use_fps=args.fps,
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    object_set_name = args.object_set.replace("/", "_")
    log_file = args.logdir / f"{args.name}_{args.scene}_{object_set_name}_{timestamp}.json"

    print(f"[icg] Config: {args.config}")
    print(f"[icg] Log: {log_file}")
    evaluator.eval_method(
        planner,
        object_count=args.object_count,
        num_runs=args.num_runs,
        num_rounds=args.num_rounds,
        num_views=args.num_views,
        static_view_fnc=sample_top_down_view if args.top_down else None,
        log_file=os.fspath(log_file),
    )


if __name__ == "__main__":
    main()
