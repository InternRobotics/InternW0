#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero1.sh <nproc_per_node> [hydra_overrides...]

Multi-node env vars:
  NNODES=2 NODE_RANK=0 MASTER_ADDR=<rank0_ip> MASTER_PORT=29500 DEEPSPEED_HOSTFILE=hostfile bash scripts/train_zero1.sh 8 --config-name <pretrain|posttrain_robotwin|posttrain_libero>
  NNODES=2 NODE_RANK=1 MASTER_ADDR=<rank0_ip> MASTER_PORT=29500 DEEPSPEED_HOSTFILE=hostfile bash scripts/train_zero1.sh 8 --config-name <pretrain|posttrain_robotwin|posttrain_libero>

For single-node launch, MASTER_ADDR/MASTER_PORT can be omitted. MASTER_PORT defaults to 29500.}"
shift

EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_HOSTFILE="${DEEPSPEED_HOSTFILE:-${DS_HOSTFILE:-${HOSTFILE:-}}}"

DEFAULT_CONFIG="pretrain"
CONFIG_NAME=""
HAS_CONFIG_NAME=0
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        CONFIG_NAME="${EXTRA_ARGS[$((i + 1))]}"
        HAS_CONFIG_NAME=1
      fi
      ;;
    --config-name=*)
      CONFIG_NAME="${arg#--config-name=}"
      HAS_CONFIG_NAME=1
      ;;
    task=*)
      echo "Error: task overrides were removed. Use --config-name pretrain|posttrain_robotwin|posttrain_libero." >&2
      exit 2
      ;;
  esac
done
if [[ "${HAS_CONFIG_NAME}" -eq 0 ]]; then
  CONFIG_NAME="${DEFAULT_CONFIG}"
fi
CONFIG_BASENAME="${CONFIG_NAME%.yaml}"

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    echo "Error: multi-node training requires RUN_ID to be set explicitly." >&2
    echo "Use the same RUN_ID on every machine, for example:" >&2
    echo "  RUN_ID=$(date +%Y-%m-%d_%H-%M-%S) NNODES=${NUM_MACHINES} NODE_RANK=0 MASTER_ADDR=${MAIN_PROCESS_IP} MASTER_PORT=${MAIN_PROCESS_PORT} bash scripts/train_zero1.sh ${NPROC_PER_NODE} --config-name ${CONFIG_BASENAME}" >&2
    echo "  RUN_ID=<same-run-id> NNODES=${NUM_MACHINES} NODE_RANK=1 MASTER_ADDR=${MAIN_PROCESS_IP} MASTER_PORT=${MAIN_PROCESS_PORT} bash scripts/train_zero1.sh ${NPROC_PER_NODE} --config-name ${CONFIG_BASENAME}" >&2
    exit 1
  fi
else
  echo "[run_id] using externally provided RUN_ID=${RUN_ID}"
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_processes=${NUM_PROCESSES} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} main_process_ip=${MAIN_PROCESS_IP} main_process_port=${MAIN_PROCESS_PORT} deepspeed_hostfile=${DEEPSPEED_HOSTFILE:-none} --config-name ${CONFIG_BASENAME} run_id=${RUN_ID}"
echo "[env] hostname=$(hostname) SLURM_NODEID=${SLURM_NODEID:-unset} SLURM_PROCID=${SLURM_PROCID:-unset} LOCAL_RANK=${LOCAL_RANK:-unset} RANK=${RANK:-unset} WORLD_SIZE=${WORLD_SIZE:-unset}"
echo "[env] NNODES=${NNODES:-unset} NODE_RANK=${NODE_RANK:-unset} MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT:-unset} HOSTFILE=${HOSTFILE:-unset} DS_HOSTFILE=${DS_HOSTFILE:-unset} DEEPSPEED_HOSTFILE=${DEEPSPEED_HOSTFILE:-unset}"
python - <<'PY'
import os
import socket

master_addr = os.environ.get("MASTER_ADDR", "")
if master_addr:
    try:
        resolved = socket.gethostbyname(master_addr)
    except OSError as exc:
        resolved = f"<resolve failed: {exc}>"
else:
    resolved = "<unset>"
print(
    "[env] resolved_master_addr="
    f"{master_addr or '<unset>'}->{resolved} local_ips={socket.gethostbyname_ex(socket.gethostname())[-1]}",
    flush=True,
)
PY

# CUDA allocator: expandable segments reduces fragmentation-induced defrag stalls.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:+${PYTORCH_CUDA_ALLOC_CONF},}expandable_segments:True"

#   "output_dir=./runs/${CONFIG_BASENAME}/${RUN_ID}" \

HYDRA_FULL_ERROR=1 accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  "${ACCELERATE_ARGS[@]}" \
  scripts/train.py --config-name "${CONFIG_NAME}" \
  "wandb.name=${CONFIG_BASENAME}" \
  "${EXTRA_ARGS[@]}"
