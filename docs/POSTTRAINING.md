# RoboTwin and LIBERO posttraining

Posttraining uses a target-domain bank with one entry. The released
RoboTwin-clean and LIBERO checkpoints are already native `.pt` files containing
the required action-side interfaces:

- action encoder;
- action decoder/head;
- K=8 soft prompt;
- proprio encoder.

The checkpoints are loaded directly.

## Common invariants

- Posttraining requires exactly one of `init_checkpoint` or `resume`.
- `num_history_frames=0`.
- `action_horizon=64`.
- `action_chunk_size=16`.
- Video plan horizon is 96 action ticks.
- Action/video stride is 8.
- RoboTwin and LIBERO use a 37D canonical action/proprio layout with explicit
  valid-slot masks.
- Short trajectories are padded and masked.
- Offset training samples one phase in `[0, max_action_offset]`; offset 0 is an
  explicit ablation.

## RoboTwin recipe

The retained reader consumes the native RoboTwin LeRobot joint-space export.
Prepare text embeddings first, then statistics:

```bash
W0_ROBOTWIN_DATA="$W0_ROBOTWIN_DATA" \
python scripts/precompute_text_embeds.py \
  --config-name posttrain_robotwin

python scripts/prepare_robotwin_stats.py \
  --dataset-root "$W0_ROBOTWIN_DATA" \
  --output assets/robotwin/normalization.json
```

The retained recipe follows the best Clean2Random run: clean demonstrations,
25 Hz (`sampling_stride=2`), every valid native window, ColorJitter, and offset
0.

Action supervision is the recorded native command at the next sampled tick.
Proprio is the current sampled state. The 14 native joint/gripper values occupy:

```text
[0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22, 33]
```

## LIBERO recipe

LIBERO remains at its native 20 Hz. Actions and proprio are row-aligned. Native
EEF10 occupies the 37D left-arm EEF slots:

```text
[7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
```

### Text cache and statistics

```bash
python scripts/prepare_libero_text_embeds.py \
  --dataset-root "$W0_LIBERO_DATA" \
  --cache-dir assets/libero/text_embeds_cache

python scripts/prepare_libero_posttrain.py \
  --dataset-root "$W0_LIBERO_DATA" \
  --output assets/libero/normalization.json
```

## Training

RoboTwin Clean2Random:

```bash
export W0_ROBOTWIN_CHECKPOINT=checkpoints/robotwin-clean.pt
bash scripts/train_zero1.sh 8 \
  --config-name posttrain_robotwin
```

The default RoboTwin config already uses `model.max_action_offset=0`.

LIBERO:

```bash
export W0_LIBERO_CHECKPOINT=checkpoints/libero.pt
bash scripts/train_zero1.sh 8 \
  --config-name posttrain_libero
```

For multi-node launch, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`,
`MASTER_PORT`, and `DEEPSPEED_HOSTFILE` as shown in `scripts/train_zero1.sh`.

## Resume semantics

- `init_checkpoint`: load model weights only; create a new optimizer and begin
  at step 0.
- `resume`: restore model, optimizer, scheduler, RNG-related training state,
  and progress when the full state is compatible.

These modes are mutually exclusive.
