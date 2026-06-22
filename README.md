# GraspingArm

Final project for the Politecnico di Torino Robotics course (01PEEYP). The
project reproduces the official ICG-Net grasping benchmark, provides a local
dataset construction and training workflow, and integrates ICG-Net with a ROS 2
robot-arm interface.

The reference paper is included as
`ICGNet_A_Unified_Approach_for_Instance-Centric_Grasping.pdf`.

## What We Implemented

- Repeatable setup of the official `icg_net` and `icg_benchmark` repositories.
- Download and validation of the official datasets and model checkpoint.
- Packed and pile benchmark evaluation with smoke and full-run entry points.
- Construction and processing of ICG-compatible grasping datasets.
- Configurable training, checkpointing, metric collection, plotting, and
  threshold analysis.
- A ROS 2 node that consumes camera point clouds and the current arm pose,
  selects a grasp target, publishes arm commands, and exposes gripper services.
- Calibrated simulation assets for the rotational gripper and clutter scenes.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `configs/` | Canonical smoke, pilot, and full training overlays |
| `scripts/` | Setup, data, training, evaluation, and reporting tools |
| `ros2_grasping/` | ROS 2 grasp-planning package and detailed guide |
| `tests/` | Dataset and training-tool tests |
| `third_party/` | Local modifications applied to the official repositories |

Downloaded repositories, datasets, checkpoints, logs, captures, and build
outputs are intentionally excluded from version control.

## Prerequisites

- Ubuntu 22.04 or 24.04, including Ubuntu under WSL 2
- An NVIDIA GPU with a working CUDA driver
- `git`, `curl`, and standard C/C++ build tools
- At least 30 GB of free disk space for dependencies, data, and outputs
- ROS 2 for the optional robot integration

## Setup

Run all commands from the repository root.

```bash
bash scripts/check_system.sh
bash scripts/fetch_icg_repos.sh
bash scripts/setup_icg_env.sh
bash scripts/download_icg_data.sh
bash scripts/check_icg_imports.sh
```

Use `bash scripts/setup_icg_env.sh --skip-apt` when system packages are already
installed or `sudo` is unavailable. The setup creates the `icg_cuda121`
environment and installs the versions required by the official repositories.

For PowerShell on Windows, the same scripts can be launched through WSL:

```powershell
.\scripts\run_from_wsl.ps1 -Script scripts/check_system.sh
.\scripts\run_from_wsl.ps1 -Script scripts/run_icg_smoke.sh
```

## Evaluation

Run a short end-to-end check:

```bash
bash scripts/run_icg_smoke.sh
```

Run the official four-run, 100-round packed and pile benchmark:

```bash
bash scripts/run_icg_full.sh
```

Summarize generated JSON logs:

```bash
python scripts/summarize_icg_logs.py logs/icg_full
```

### First Successful Full Run

The first complete official-checkpoint evaluation used four runs of 100 rounds
per scene.

| Scene | Local GSR (%) | Paper GSR (%) | Local DR (%) | Paper DR (%) |
| --- | ---: | ---: | ---: | ---: |
| packed | 98.63 +/- 0.69 | 97.7 +/- 0.9 | 98.22 +/- 0.87 | 97.5 +/- 0.3 |
| pile | 91.19 +/- 1.21 | 92.0 +/- 2.6 | 91.56 +/- 1.85 | 94.1 +/- 1.4 |

### Best Recorded Scene Results

Each row reports the strongest complete run recorded for that scene.

| Scene | GSR (%) | DR (%) |
| --- | ---: | ---: |
| packed | 98.63 +/- 0.69 | 98.22 +/- 0.87 |
| pile | 91.44 +/- 0.46 | 92.22 +/- 1.43 |

## Dataset Construction

The dataset tool supports canonical packed/pile generation, individual shard
processing, local audits, and optional object-package extensions.

Generate and process a small canonical dataset:

```bash
python scripts/build_icg_dataset.py canonical --stage pilot
```

Generate the full canonical shard layout:

```bash
python scripts/build_icg_dataset.py canonical --stage full
```

Process an existing raw shard:

```bash
python scripts/build_icg_dataset.py process-shard \
  --raw-root data/raw/example \
  --output-root data/processed/example
```

Create reconstruction and target-statistics reports:

```bash
python scripts/build_icg_dataset.py audit
python scripts/audit_icg_reconstructed_targets.py
```

Generated data is stored under `data/` by default. Set `ICG_DATA_ROOT` or pass
`--data-root` to use another location.

## Training

The five retained overlays cover synthetic verification, reconstructed-data
pilot runs, and the complete reconstructed shard layout:

- `configs/train_icg_smoke.yaml`
- `configs/train_icg_official_example.yaml`
- `configs/train_icg_reconstructed_pilot_smoke.yaml`
- `configs/train_icg_reconstructed_pilot_official.yaml`
- `configs/train_icg_reconstructed_full_example.yaml`

Run the synthetic training check:

```bash
python scripts/train_icg.py \
  --config third_party/icg_benchmark/data/icgnet/51--0.656/config.yaml \
  --overlay configs/train_icg_smoke.yaml \
  --device cpu \
  --max-epochs 1
```

Train on the reconstructed pilot dataset with CUDA:

```bash
python scripts/train_icg.py \
  --config third_party/icg_benchmark/data/icgnet/51--0.656/config.yaml \
  --overlay configs/train_icg_reconstructed_pilot_official.yaml \
  --device cuda
```

Generate plots and the benchmark comparison for a completed run:

```bash
python scripts/plot_icg_training_metrics.py logs/training/<run-directory>
```

Evaluate a local checkpoint or sweep its grasp threshold:

```bash
python scripts/run_icg_eval.py --checkpoint <checkpoint> --scene packed
python scripts/sweep_icg_grasp_thresholds.py \
  --config third_party/icg_benchmark/data/icgnet/51--0.656/config.yaml \
  --overlay configs/train_icg_reconstructed_pilot_official.yaml \
  --checkpoint <checkpoint>
```

## ROS 2 Integration

Build and run the package after the ICG environment and ROS 2 installation are
available:

```bash
colcon build --packages-select ros2_grasping
source install/setup.bash
ros2 run ros2_grasping grasp_planner_node
```

The node defaults to the calibrated fixed-camera simulation pose. It can also
derive the camera pose from the current end-effector pose. Topics, services,
parameters, calibration values, and example commands are documented in
[`ros2_grasping/README.md`](ros2_grasping/README.md).

## Verification

With the `icg_cuda121` environment active:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q scripts ros2_grasping
for script in scripts/*.sh; do bash -n "$script"; done
colcon build --packages-select ros2_grasping
```
