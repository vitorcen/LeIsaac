# LeIsaac fine-tune scripts

通用 LeRobot 策略微调脚本。任何能被 `lerobot-train --policy.path=...` 加载的模型（SmolVLA / ACT / pi0 / DreamZero / 自家 fork）都可以走这套流程。

## 前置

```bash
# 1) 下载数据集
bash datasets/download.sh <ORG>/<DATASET>

# 2) 转 v3.0（如果是 v2.1 数据集）
bash datasets/convert_to_v30.sh <ORG>/<DATASET>
```

详见 [`datasets/README.md`](../../datasets/README.md)。

## 通用入口

```bash
bash scripts/finetune/lerobot_finetune.sh
```

行为完全由环境变量驱动；下表是全部 knob：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `BASE_MODEL` | `lerobot/smolvla_base` | `--policy.path` 值（HF repo 或本地目录）|
| `DATASET_REPO_ID` | `LightwheelAI/leisaac-pick-orange` | `--dataset.repo_id` |
| `DATASET_ROOT` | `datasets/raw/<basename>` | 本地 v3.0 数据路径 |
| `OUTPUT_NAME` | `<base>-<dataset>` | 输出子目录名（位于 `outputs/`）|
| `STEPS` | `20000` | 训练步数 |
| `BATCH_SIZE` | `64` | per-device batch |
| `NUM_WORKERS` | `4` | dataloader workers |
| `SAVE_FREQ` | `5000` | ckpt 保存间隔 |
| `RENAME_MAP` | (空) | JSON dict：sim 键 → policy 期望键。仅当 base 模型不用自然键时需要 |
| `EXTRA_ARGS` | (空) | 透传给 `lerobot-train` 的额外 flag |
| `CONDA_ENV` | `lerobot` | conda env 名 |

最终 ckpt：`outputs/<OUTPUT_NAME>/checkpoints/last/pretrained_model`。

## 已验证示例

### SmolVLA2（基于 `lerobot/smolvla_base`）

`smolvla_base` 的 saved config.json 声明了 3 个占位相机 `camera1/2/3 @ 256x256`，draccus 对 dict 做 **合并** 而非替换，所以 `--policy.input_features=...` CLI override 无法清掉这 3 个占位槽（实测会变成 5 个槽：3 占位 + 2 自然，camera3 没数据训练→死权重，且 256x256 损失分辨率）。

**正确做法**：先用 `prepare_smolvla_base.sh` 把 base 克隆到本地、清掉 `input_features` 和 `empty_cameras`，再让 `lerobot-train` 从数据集 features 自动生成（自然键 + 真实 480×640）。

```bash
# 1) 准备 schema-free 的本地 base（首次一次性，~865MB）
bash scripts/finetune/prepare_smolvla_base.sh
# → outputs/.bases/smolvla_base_no_features/

# 2) 微调
BASE_MODEL=$(pwd)/outputs/.bases/smolvla_base_no_features \
DATASET_REPO_ID=LightwheelAI/leisaac-pick-orange \
OUTPUT_NAME=smolvla2-leisaac-pick-orange \
STEPS=30000 \
BATCH_SIZE=8 \
NUM_WORKERS=2 \
SAVE_FREQ=5000 \
EXTRA_ARGS='--dataset.video_backend=pyav' \
bash scripts/finetune/lerobot_finetune.sh
```

**已验证设置**：
- `BATCH_SIZE=8`：edge-inference 公开 ckpt 使用的值；4090 单卡 ~7GB VRAM，~9 step/s（vs batch=64 时 1.2 step/s，因为大 batch 撞 PCIe/IO 瓶颈）。
- `STEPS=30000`：edge-inference 使用的步数（~9 step/s × 30k = 56min on 4090）。
- `--dataset.video_backend=pyav`：torchcodec + 多 worker 在长跑时会 segfault；pyav 慢一点但稳。
- 不传 `RENAME_MAP`：base 被剥光后由数据集自然键自动生成 schema。

### ACT（基于 `lerobot/act`，假设官方有对应基座）

ACT 类模型用 LeRobot 数据集的自然键，无需 rename。

```bash
BASE_MODEL=<ACT_BASE_REPO> \
DATASET_REPO_ID=LightwheelAI/leisaac-pick-orange \
OUTPUT_NAME=act-leisaac-pick-orange \
STEPS=100000 \
BATCH_SIZE=8 \
bash scripts/finetune/lerobot_finetune.sh
```

### DreamZero（占位 — 等模型上线）

```bash
BASE_MODEL=<DreamZero_REPO> \
DATASET_REPO_ID=LightwheelAI/leisaac-pick-orange \
OUTPUT_NAME=dreamzero-leisaac-pick-orange \
STEPS=30000 \
BATCH_SIZE=32 \
RENAME_MAP='<根据 DreamZero policy config 的相机键填>' \
EXTRA_ARGS='--policy.freeze_vision_encoder=true' \
bash scripts/finetune/lerobot_finetune.sh
```

## 推理验证

训完用 `policy_inference.py` 跑仿真。模型类型 `lerobot-<model_type>`（由 ckpt 的 `config.json` 的 `type` 字段决定）：

```bash
# 1) 启动 LeRobot policy_server（端口 8080）
bash ~/work/isaaclab-experience/scripts/policy_server.sh start lerobot

# 2) 启动 Isaac Sim 客户端
cd ~/work/isaaclab-experience/LeIsaac && \
  PYTHONUNBUFFERED=1 python -u scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 --episode_length_s=120 \
    --policy_type=lerobot-smolvla \
    --policy_host=127.0.0.1 --policy_port=8080 \
    --policy_timeout_ms=15000 \
    --policy_language_instruction='Pick the orange to the plate' \
    --policy_checkpoint_path=/path/to/outputs/<OUTPUT_NAME>/checkpoints/last/pretrained_model \
    --policy_action_horizon=16 --device=cuda --enable_cameras
```

LeIsaac 客户端的 `_build_camera_feature_map(ckpt_path, sim_cameras)` 会读 ckpt 的 `config.json/input_features`，自然键命中时返回 `None`，否则自动构造 rename map——所以推理 client 端不需要再传 RENAME_MAP。
