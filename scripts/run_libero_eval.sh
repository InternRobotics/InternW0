#!/usr/bin/env bash
# Two-process LIBERO evaluation: a w0 policy server and a LIBERO environment client.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT=${CKPT:?Set CKPT to a w0 LIBERO posttraining checkpoint}
SUITE=${SUITE:-libero_goal}
TASK_ID=${TASK_ID:-0}
NUM_TRIALS=${NUM_TRIALS:-50}

# A checkpoint trained with 16-step chunks is not equivalent to a chunk-8
# checkpoint merely because only eight actions are executed before replanning.
EXECUTE_N_ACTIONS=${EXECUTE_N_ACTIONS:-16}
if [[ -z "${MAX_STEPS:-}" ]]; then
  case "$SUITE" in
    libero_spatial) MAX_STEPS=220 ;;
    libero_object) MAX_STEPS=280 ;;
    libero_goal) MAX_STEPS=300 ;;
    libero_10) MAX_STEPS=520 ;;
    *) echo "Unknown LIBERO suite: $SUITE" >&2; exit 2 ;;
  esac
fi

PORT=${PORT:-8765}
POLICY_GPU=${POLICY_GPU:-0}
RENDER_GPU=${RENDER_GPU:-1}
LIBERO_PATH=${LIBERO_PATH:-external/LIBERO}
LIBERO_PYTHON=${LIBERO_PYTHON:-python}
W0_PYTHON=${W0_PYTHON:-python}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/libero/evaluation}
LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-.cache/libero_config}
W0_WAN_MODEL_DIR=${W0_WAN_MODEL_DIR:?Set W0_WAN_MODEL_DIR to the Wan2.2 model directory}
if [[ "$W0_WAN_MODEL_DIR" != /* ]]; then
  echo "Error: W0_WAN_MODEL_DIR must be an absolute path to the local Wan2.2 model directory." >&2
  exit 2
fi

export LIBERO_PATH
export W0_LIBERO_CHECKPOINT="$CKPT"
export LIBERO_CONFIG_PATH
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$W0_WAN_MODEL_DIR"
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}
export W0_LIBERO_IMAGE_CODEC=${W0_LIBERO_IMAGE_CODEC:-PNG}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="$POLICY_GPU"

mkdir -p "$OUTPUT_DIR" "$LIBERO_CONFIG_PATH"
serve_args=(--ckpt "$CKPT" --port "$PORT")
if [[ "$EXECUTE_N_ACTIONS" -gt 0 ]]; then
  serve_args+=(--execute-n-actions "$EXECUTE_N_ACTIONS")
fi
if [[ -n "${POLICY_SEED:-}" ]]; then
  serve_args+=(--seed "$POLICY_SEED")
fi

"$W0_PYTHON" "$ROOT/experiments/libero/serve_libero_policy.py" \
  "${serve_args[@]}" &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT

for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null

export CUDA_VISIBLE_DEVICES="$RENDER_GPU"
export PYTHONPATH="$LIBERO_PATH:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${MUJOCO_OVERLAY:-}" ]]; then
  export PYTHONPATH="$MUJOCO_OVERLAY:$PYTHONPATH"
fi
if [[ -n "${NVGL_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$NVGL_DIR:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${EGL_VENDOR_LIBRARY_DIRS:-}" ]]; then
  export __EGL_VENDOR_LIBRARY_DIRS="$EGL_VENDOR_LIBRARY_DIRS"
fi
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_EGL_DEVICE_ID="$RENDER_GPU"

video_args=()
if [[ "${SAVE_VIDEO:-0}" == "1" ]]; then
  video_args+=(--save-video --video-fps "${VIDEO_FPS:-20}")
fi

"$LIBERO_PYTHON" "$ROOT/experiments/libero/eval_libero_client.py" \
  --suite "$SUITE" \
  --task-id "$TASK_ID" \
  --num-trials "$NUM_TRIALS" \
  --seed 7 \
  --settle-steps 10 \
  --binarize-gripper \
  --max-steps "$MAX_STEPS" \
  --port "$PORT" \
  --output-dir "$OUTPUT_DIR" \
  "${video_args[@]}"
