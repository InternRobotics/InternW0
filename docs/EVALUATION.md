# Evaluation

Only RoboTwin and LIBERO evaluation adapters are included in this release.

## RoboTwin

RoboTwin is an external dependency. Set `EVALUATION.robotwin_root` to a checkout
that contains its standard `policy` directory.

Single-task example:

```bash
python experiments/robotwin/eval_robotwin_single.py \
  EVALUATION.robotwin_root=external/RoboTwin \
  EVALUATION.task_name=<task> \
  EVALUATION.ckpt=<checkpoint.pt>
```

The fixed-clock adapter submits joint targets at the trained control period
instead of relying on an unrelated controller integration schedule.
By default it resolves the released posttraining recipe from `eval_robotwin`;
set `EVALUATION.training_config` only when you need a run-specific resolved YAML.

## LIBERO

The LIBERO runner starts two processes:

1. a w0 policy server;
2. a client linked against the external LIBERO environment.

Required inputs:

- `CKPT`: w0 LIBERO posttraining checkpoint;
- `W0_WAN_MODEL_DIR`: Wan2.2 model directory;
- `W0_LIBERO_STATS`: matching normalization file;
- `W0_LIBERO_TEXT_CACHE`: matching UMT5 cache;
- `LIBERO_PATH`: external LIBERO checkout.

Example:

```bash
CKPT=assets/libero/checkpoint.pt \
W0_WAN_MODEL_DIR=/absolute/path/to/Wan2.2-TI2V-5B \
W0_LIBERO_STATS=assets/libero/normalization.json \
W0_LIBERO_TEXT_CACHE=assets/libero/text_embeds_cache \
LIBERO_PATH=external/LIBERO \
SUITE=libero_goal TASK_ID=0 NUM_TRIALS=50 \
bash scripts/run_libero_eval.sh
```

The default evaluation executes a complete 16-action chunk unless
`EXECUTE_N_ACTIONS` is explicitly overridden. Changing this value changes the
closed-loop protocol and is not equivalent to training a different chunk-size
model.

## Reporting

A reproducible evaluation record should include:

- checkpoint path and step;
- Git commit or source snapshot hash;
- dataset/statistics fingerprint;
- suite and task IDs;
- environment seed and policy seed;
- number of executed actions per replan;
- inference steps;
- controller period;
- simulation version;
- camera transform and image codec;
- success criterion and trial count.
