<p align="left">
  <img src="assets/li-auto.svg" alt="Li Auto" width="150">
</p>

<h1 align="center">
  <img src="assets/Logo.svg" alt="ME-U0" width="320">
</h1>

<p align="center">
  MachEmbodied-U0: Unified Understanding and Generation Model for Embodied Intelligence
</p>

<p align="center">
  <a href="https://raw.githubusercontent.com/MachEmbodied/ME-U0/main/assets/ME_U0.pdf">📄 Technical Report</a> &nbsp;|&nbsp;
  <a href="https://machembodied.com/ME-U/ME-U0.html">🌐 Project Page</a>
</p>

---

ME-U0 connects task-grounded understanding with joint visual dynamics and action generation through a Mixture-of-Transformers architecture. This repository provides post-training and evaluation code for LIBERO and RoboDojo.

<p align="center">
  <img src="assets/fig1-framework.png" alt="ME-U0 framework: task-grounded understanding, visual dynamics, and robot action generation" width="100%">
</p>

## Model Architecture

The understanding expert predicts subtasks and affordances, while the generation expert jointly predicts future visual observations and robot actions. Multi-rate Rotary Position Encoding (MRPE) aligns visual dynamics with fine-grained control.

<p align="center">
  <img src="assets/fig6-architecture.png" alt="ME-U0 architecture with understanding and generation experts, shared multimodal attention, and MRPE" width="100%">
</p>

## Getting Started

Install the dependencies in a compatible PyTorch accelerator environment:

```bash
bash scripts/install.sh
```

Prepare the Lance backbone assets (`Lance_3B_Video/`, `Qwen2.5-VL-ViT/`, and `Wan2.2_VAE.pth`) under one directory, and a compatible ME-U0 pretraining checkpoint:

```bash
export LEAP_MODEL_ROOT=/path/to/lance-assets
export ME_U0_PRETRAINED_PTH=/path/to/checkpoints/step_N
export LEAP_WORK_ROOT=/path/to/work_dirs
```

Model weights, datasets, and simulator assets are not bundled. Set the dataset paths in the corresponding YAML under `leap/configs/data/` before training.

| Recipe | Configuration | Action horizon | Video stride |
| --- | --- | ---: | ---: |
| LIBERO | [`libero_posttraining.yaml`](leap/configs/experiments/libero_posttraining.yaml) | 24 | 3 |
| RoboDojo | [`robodojo_sim_posttraining.yaml`](leap/configs/experiments/robodojo_sim_posttraining.yaml) | 48 | 2 |

LIBERO uses native delta-EEF actions and the dataset's min/max statistics. RoboDojo uses chunk-start delta-joint actions and the included H48 q01/q99 statistics.

### Post-training

Run from the repository root after setting the model and dataset paths above.
`ME_U0_PRETRAINED_PTH` loads model weights; `--resume latest` restores an existing training run, including optimizer state.

**GPU training (one node, eight GPUs):**

```bash
bash scripts/ME_U0/run_multinode.sh 1 \
  --nproc-per-node 8 \
  --exp-name libero_posttraining \
  --work-root "$LEAP_WORK_ROOT" \
  --config leap/configs/experiments/libero_posttraining.yaml
```

**PPU training:** install the vendor runtime first, then use:

```bash
MASTER_PORT=29900 bash scripts/ME_U0/train_ppu.sh \
  --nnodes 1 --nproc-per-node 16 \
  --shared-repo "$PWD" --exp-name libero_posttraining \
  --work-root "$LEAP_WORK_ROOT" \
  --train-config leap/configs/experiments/libero_posttraining.yaml
```

For RoboDojo, replace the config with `robodojo_sim_posttraining.yaml` and choose a separate experiment name. For multi-node training, launch on every node with the same shared work root, `MASTER_ADDR`, `MASTER_PORT`, and node count, and a unique `NODE_RANK` (`0` to `N-1`).

### Evaluation

The policy and simulator use separate environments. `scripts/install.sh` installs the policy dependencies; prepare the simulator environments and assets separately:

```bash
export LIBERO_UNIFIED_ROOT=/path/to/unified_eval
export ROBODOJO_UNIFIED_ROOT=/path/to/unified_eval
```

The default layout is:

```text
unified_eval/
├── conda/libero_py38/bin/python
├── source/LIBERO/
├── source/LIBERO-plus/
├── source/RoboDojo-25691aa78fb3/
└── scripts/run_robodojo_env.sh
```

LIBERO and LIBERO-Plus share the simulator Python environment. For another layout, set `CONDA_LIBERO_PY` and `LIBERO_HOME` (the corresponding LIBERO or LIBERO-Plus source tree); for RoboDojo, set `ROBODOJO_ROOT`. Configure each simulator's asset paths for your installation.

**LIBERO:**

```bash
bash scripts/ME_U0/eval_libero.sh \
  --config leap/configs/experiments/libero_posttraining.yaml \
  --checkpoint /path/to/libero/checkpoints/step_25000 \
  --output-dir /path/to/eval_libero \
  --task-suite all --server-gpu 0,1,2,3 --sim-gpus 0,1,2,3 \
  --image-size 256 --num-inference-steps 12 --action-chunk-size 24
```

**LIBERO-Plus:** evaluate the same LIBERO-trained checkpoint with LIBERO normalization statistics.

```bash
ME_U0_LIBERO_PLUS_NORMALIZER_SOURCE=libero \
bash scripts/ME_U0/eval_libero_plus_distributed.sh \
  --config leap/configs/experiments/libero_posttraining.yaml \
  --checkpoint /path/to/libero/checkpoints/step_25000 \
  --output-root /path/to/eval_libero_plus \
  --task-suite all --libero-plus-shards-per-suite 4 \
  --num-nodes 1 --node-rank 0 \
  --server-gpus 0,1,2,3 --sim-gpus 0,1,2,3 \
  --image-size 256 --num-inference-steps 12 --action-chunk-size 24
```

**RoboDojo:**

```bash
bash scripts/ME_U0/eval_robodojo_distributed.sh \
  --config leap/configs/experiments/robodojo_sim_posttraining.yaml \
  --checkpoint /path/to/robodojo/checkpoints/step_30000 \
  --output-root /path/to/eval_robodojo \
  --task-suite all --seeds 0,1,2 --eval-num native \
  --num-nodes 1 --node-rank 0 \
  --server-gpus 0,1,2,3 --sim-gpus 0,1,2,3 \
  --num-inference-steps 12 --action-chunk-size 36
```

For distributed evaluation, run on each machine with the same `--num-nodes` and shared `--output-root`, and a unique `--node-rank`. Use a separate output directory for each experiment.

Camera layout and resize sizes come from the training config. `--action-chunk-size` controls how many predicted actions are executed before replanning; it does not change the trained action horizon. Run any entry point with `--help` for additional options.
