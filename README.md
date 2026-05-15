# LeIsaac — vitorcen fork

> Fork of **[LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac)** (Apache-2.0). This fork extends the upstream project with a reusable fine-tuning scaffold, a SO-101 PickOrange multi-policy comparison study, and a few client-side fixes needed to evaluate non-trivial VLAs through Isaac Sim.

- **Upstream**: https://github.com/LightwheelAI/leisaac
- **Upstream docs**: https://lightwheelai.github.io/leisaac/
- **This fork**: https://github.com/vitorcen/LeIsaac

The original LeIsaac repo provides SO-101 teleoperation in Isaac Sim plus the GR00T N1.5 / N1.6 fine-tuning recipes for the `LeIsaac-SO101-PickOrange-v0` task. For original features, follow [upstream docs](https://lightwheelai.github.io/leisaac/). The rest of this README describes **what this fork adds on top**.

---

## What this fork adds

### 1. Reusable LeRobot fine-tune scaffold

End-to-end, env-driven scripts for downloading a LeRobot dataset, converting v2.1 → v3.0, and running `lerobot-train` with sane defaults. Same scaffold targets SmolVLA / ACT / Diffusion Policy / DiT / and (with one extra step) future models.

| Script | Purpose |
| --- | --- |
| [`datasets/download.sh`](datasets/download.sh) | `bash datasets/download.sh <ORG>/<DATASET>` — pulls any LeRobot dataset into `datasets/raw/<basename>/` |
| [`datasets/convert_to_v30.sh`](datasets/convert_to_v30.sh) | In-place v2.1 → v3.0 conversion (lerobot ≥ 0.5.x requires v3.0); idempotent |
| [`scripts/finetune/lerobot_finetune.sh`](scripts/finetune/lerobot_finetune.sh) | Generic `lerobot-train` wrapper, all knobs as env vars (`BASE_MODEL` / `DATASET_REPO_ID` / `STEPS` / `BATCH_SIZE` / `RENAME_MAP` / `EXTRA_ARGS` / ...) |
| [`scripts/finetune/prepare_smolvla_base.sh`](scripts/finetune/prepare_smolvla_base.sh) | One-shot clone of `lerobot/smolvla_base` with `input_features` + `empty_cameras` stripped — needed because draccus' CLI override merges dicts instead of replacing, leaving leftover `camera1/2/3 @ 256×256` placeholder slots |

Workflow docs:
- [`datasets/README.md`](datasets/README.md)
- [`scripts/finetune/README.md`](scripts/finetune/README.md)

### 2. PickOrange policy comparison study

Treating `LeIsaac-SO101-PickOrange-v0` as a benchmark and running multiple policy families through the same evaluation harness. Results (3 episodes × 60s unless noted):

| Policy | Params | `config.type` | Result | Notes |
| --- | --- | --- | --- | --- |
| [`shadowHokage/act_policy`](https://huggingface.co/shadowHokage/act_policy) | ~80M | `act` | ✅ succeeds within 60s | from-scratch, 10k step, batch 8 |
| [`LightwheelAI/leisaac-pick-orange-v0`](https://huggingface.co/LightwheelAI/leisaac-pick-orange-v0) | ~3B | `gr00t_n1_5` | ✅ 1/1 within 60s | upstream official ckpt |
| [`edge-inference/smolvla-so101-pick-orange`](https://huggingface.co/edge-inference/smolvla-so101-pick-orange) | ~450M | `smolvla` | ❌ 0/3 @ 60s | third-party SmolVLA v1 fine-tune |
| local `smolvla2-leisaac-pick-orange` *(misnomer — actually `config.type=smolvla` v1)* | ~450M | `smolvla` | 🟡 2/5 @ horizon=50, 5×120s | 30k step, batch 8, schema-free base |

**Headline**: at 60 episodes × 36k frames data scale and 60s episode budget, small + visuomotor-specialised models (ACT) outperform larger VLA models (SmolVLA family).

Full design + priorities for the next round of policy work (Diffusion Policy → DiT → SmolVLA2 → Octo → RDT) is in [`docs/finetune/policy_comparison_priorities.html`](docs/finetune/policy_comparison_priorities.html).

### 3. Design docs

- [`docs/finetune/smolvla2_finetune_pick_orange.html`](docs/finetune/smolvla2_finetune_pick_orange.html) — design + post-mortem of the SmolVLA fine-tune attempt on PickOrange (v1 fail, v2 partial, schema-free base recipe)
- [`docs/finetune/policy_comparison_priorities.html`](docs/finetune/policy_comparison_priorities.html) — comparative study + prioritised plan for Diffusion Policy / DiT Policy / SmolVLA2 (the real one, at [huggingface/VLAb](https://github.com/huggingface/VLAb)) / Octo / RDT-1B

### 4. LeIsaac client fixes for non-trivial VLAs

These changes live in `source/leisaac/leisaac/policy/service_policy_clients.py` and `scripts/evaluation/policy_inference.py`. Upstream LeIsaac only validates against GR00T-style policies; SmolVLA exposed several edge cases:

- **Auto camera schema mapping** — `_build_camera_feature_map(ckpt_path, sim_cameras)` reads the ckpt's `config.json` and returns `(rename_map, empty_camera_feats)`. Naturally-keyed ckpts (front/wrist) → no rename; placeholder-keyed ckpts (camera1/2/3) → positional rename + zero-pad for unused slots. Avoids `KeyError: 'camera1'` on first inference.
- **`must_go=True` on every observation** — bypasses the server's "Observation too similar to last obs predicted" dedup filter, which otherwise drops valid observations whenever the sim hasn't moved between calls.
- **Bounded retry without deadlock** — `_receive_action()` retries 8× (200ms cap) for slow first inference; removed the `skip_send_observation` flag that previously deadlocked the client forever after a single retry exhaustion.
- **Server-state hygiene rules** — `policy_server` keeps `last_processed_obs` across client sessions, so a failed client run will starve the next one. Always `kill -9` the server + restart before re-evaluating. Same for Isaac Sim processes that fail to release GPU memory after a Python exception.

---

## Quick start (this fork)

```bash
# 1) Get the dataset (LeIsaac SO-101 PickOrange, ~670 MB)
bash datasets/download.sh
bash datasets/convert_to_v30.sh

# 2) Fine-tune SmolVLA (with the schema-free base hack)
bash scripts/finetune/prepare_smolvla_base.sh
BASE_MODEL=$(pwd)/outputs/.bases/smolvla_base_no_features \
DATASET_REPO_ID=LightwheelAI/leisaac-pick-orange \
OUTPUT_NAME=smolvla2-leisaac-pick-orange \
STEPS=30000 BATCH_SIZE=8 NUM_WORKERS=2 SAVE_FREQ=5000 \
EXTRA_ARGS='--dataset.video_backend=pyav' \
bash scripts/finetune/lerobot_finetune.sh

# 3) Evaluate in Isaac Sim
bash ~/work/isaaclab-experience/scripts/policy_server.sh start lerobot
cd ~/work/isaaclab-experience/LeIsaac && \
  PYTHONUNBUFFERED=1 python -u scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=3 --episode_length_s=60 \
    --policy_type=lerobot-smolvla \
    --policy_host=127.0.0.1 --policy_port=8080 \
    --policy_timeout_ms=15000 \
    --policy_language_instruction='Pick the orange to the plate' \
    --policy_checkpoint_path=$(pwd)/outputs/smolvla2-leisaac-pick-orange/checkpoints/last/pretrained_model \
    --policy_action_horizon=50 --device=cuda --enable_cameras
```

For upstream features (teleoperation, datagen state machine, GR00T fine-tuning, etc.) follow the [upstream docs](https://lightwheelai.github.io/leisaac/).

---

## Acknowledgements

This fork builds on [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac), which itself acknowledges [IsaacLab](https://github.com/isaac-sim/IsaacLab) and [LeRobot](https://github.com/huggingface/lerobot). All upstream contributors retain their attribution; this fork's changes are additive (new scripts/docs) and a small patch to the LeRobot service client.

## Citation

Cite the upstream project per their convention:

```txt
@software{Lightwheel_and_LeIsaac_Project_Developers_LeIsaac_2025,
  author = {{Lightwheel} and {LeIsaac Project Developers}},
  license = {Apache-2.0},
  title = {{LeIsaac}},
  url = {https://github.com/LightwheelAI/leisaac},
  version = {0.4.0},
  year = {2026}
}
```

## License

Apache-2.0, same as upstream. See [LICENSE](LICENSE).
