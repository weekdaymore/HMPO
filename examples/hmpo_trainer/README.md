# HMPO Trainer

Reference implementation of **HMPO** (Hybrid Median-length Policy Optimization for Chain-of-Thought Compression),
built on top of [verl](https://github.com/volcengine/verl) (v0.7.1) using the
`experimental/reward_loop` architecture.

In the length reward, HMPO uses the **median length of correct rollouts** within
each prompt group as the baseline `b`, and applies a cosine length reward to
guide the model toward shorter reasoning while preserving accuracy.

## Layout

```
examples/hmpo_trainer/
├── README.md
├── requirements.txt
├── run_hmpo_qwen3_5_median_lda0.8_multi.sh      # HMPO training entrypoint
└── dataset/
    ├── DeepMath-103K-v1_with_token_length_5_9.parquet   # training set (6977 rows)
    └── aime-2024.parquet                                # validation set (AIME'24)
```

## Environment Setup

The code is based on verl v0.7.1 and is verified with **CUDA 12.9 / Python 3.12**.
We recommend a fresh conda environment.

```bash
# 1) create & activate the environment
conda create -n hmpo python==3.12 -y
conda activate hmpo

# 2) install PyTorch (CUDA 12.9 build)
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# 3) install verl itself (from the repo root)
cd <repo_root>          # the verl root that contains this examples/ dir
pip install -e .

# 4) install the HMPO extra requirements
pip install -r examples/hmpo_trainer/requirements.txt

# 5) install the vendored mbridge (Megatron <-> HuggingFace bridge, Qwen3.5 support)
pip install -e third_party/mbridge

# 6) install flash-attn (build against the installed torch/CUDA; takes a while)
pip install flash-attn==2.8.1 --no-build-isolation
```

### About vLLM

The rollout engine uses a custom vLLM build with Qwen3.5 support
(`vllm==0.1.dev1+g0de533398`). A plain `pip install vllm` will **not** provide
Qwen3.5 support. Use the vLLM build that ships with your Qwen3.5-capable image,
or build from the corresponding vLLM commit. The two patched files under
`verl/third_party/vllm/` and `verl/workers/rollout/vllm_rollout/` adapt verl to
this build.

> **Version note:** `transformers` **must be `5.3.0`**. Earlier versions
> (e.g. 5.2.x) raise `TypeError: unsupported operand type(s) for |: 'list' and 'set'`
> when loading the Qwen3.5 rope config.

## Data

The training/validation parquet files are provided under `dataset/`:

| File | Rows | `data_source` | Role |
|------|------|---------------|------|
| `DeepMath-103K-v1_with_token_length_5_9.parquet` | 6977 | `DeepMath-103K` | train |
| `aime-2024.parquet` | 960 | `math_dapo` | validation (AIME'24) |

Each row follows the standard verl RL schema (`prompt`, `reward_model`,
`data_source`, `extra_info`, ...). The `DeepMath-103K` / AIME data sources are
registered in `verl/utils/reward_score/__init__.py`.

## Training

Set the model path (a HuggingFace-format Qwen3.5 checkpoint), then launch:

```bash
export MODEL_PATH=/your/path/to/Qwen3.5-9B
# Data defaults to ./dataset/; override with TRAIN_FILE / TEST_FILE if needed.
# Model size and parallelism (TP/PP/GEN_TP) are controlled inside the script.

bash examples/hmpo_trainer/run_hmpo_qwen3_5_median_lda0.8_multi.sh
```

### Multi-node launch

Each script forms a Ray cluster via the `RANK / NODE_NUM / MASTER_ADDR / MASTER_PORT`
environment variables:

- `RANK=0`: starts the Ray head and submits the training job
- other ranks: join the cluster as Ray workers

## Custom Components (shipped with this repo)

| Component | Location |
|-----------|----------|
| reward manager `hmpo` | `verl/experimental/reward_loop/reward_manager/hmpo.py` |
| reward manager registration | `verl/experimental/reward_loop/reward_manager/__init__.py` |
| data source registration (DeepMath-103K / AIME) | `verl/utils/reward_score/__init__.py` |
| in-training per-group batch median branch | `verl/trainer/ppo/ray_trainer.py` |
| vendored Megatron bridge | `third_party/mbridge/` |

## Key Hyperparameters

| Argument | Value | Meaning |
|----------|-------|---------|
| `reward_model.reward_manager` | `hmpo` | select the HMPO reward manager |
| `overlong_buffer_cfg.multi` | `True` | `reward = length_reward * acc` (vs. additive) |
| `overlong_buffer_cfg.lambda` | `0.8` | cosine length-reward bias |
| `overlong_buffer_cfg.len` | `8192` | fallback baseline when a group has no correct rollout |
| `reward_model.enable` | `True` | route reward through the local batch path |
| `reward_model.enable_resource_pool` | `False` | compute reward locally (needs whole-group median) |
