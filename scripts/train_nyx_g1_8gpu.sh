#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

export PATH="$REPO_DIR/.venv/bin:$PATH"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=disable
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$REPO_DIR/outputs/cache"
export TORCHINDUCTOR_CACHE_DIR="$REPO_DIR/outputs/cache/torchinductor"
export TORCHINDUCTOR_COMPILE_THREADS=1

MOTION_SUBSET="${MORPHMIMIC_SUBSET_YAML:-$REPO_DIR/configs/data/nyx_g1_all.yaml}"
TRAINING_RUN="${MORPHMIMIC_RUN_NAME:-nyx_g1_kp_full_8gpu_100k}"

TRAIN_ARGS=( \
  --robot g1 \
  --subset-yaml "$MOTION_SUBSET" \
  --gpu-ids 0 1 2 3 4 5 6 7 \
  --num-envs 4096 \
  --iterations 100000 \
  --lr-decay-iters 100000 \
  --kp-interface \
  --kp-masking full \
  --kp-env-frac 0.5 \
  --library-dtype float32 \
  --compile default \
  --ckpt-every 5000 \
  --log-every 25 \
  --log-dir "$REPO_DIR/logs" \
  --run-name "$TRAINING_RUN" \
  "$@" \
)

if [[ "${MORPHMIMIC_LAUNCHER:-torchrunx}" == "torchrun" ]]; then
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
  MORPHMIMIC_MASTER_ADDR="${MORPHMIMIC_MASTER_ADDR:-$("$REPO_DIR/.venv/bin/python" -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')}"
  exec "$REPO_DIR/.venv/bin/python" -u -m torch.distributed.run \
    --nnodes 1 --nproc-per-node 8 \
    --master-addr "$MORPHMIMIC_MASTER_ADDR" \
    --master-port "${MORPHMIMIC_MASTER_PORT:-29537}" \
    --log-dir "$REPO_DIR/logs/$TRAINING_RUN/torchrun" \
    --redirects 3 --tee 0:3 --module morphmimic.scripts.train "${TRAIN_ARGS[@]}"
else
  exec "$REPO_DIR/.venv/bin/python" -u -m morphmimic.scripts.train "${TRAIN_ARGS[@]}"
fi
