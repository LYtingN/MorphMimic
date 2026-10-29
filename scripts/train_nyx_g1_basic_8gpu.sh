#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MORPHMIMIC_SUBSET_YAML="$REPO_DIR/configs/data/nyx_g1_basic.yaml"
export MORPHMIMIC_RUN_NAME="${MORPHMIMIC_RUN_NAME:-nyx_g1_basic_8gpu_100k_$(date -u +%Y%m%d_%H%M%S)}"
export MORPHMIMIC_LAUNCHER="${MORPHMIMIC_LAUNCHER:-torchrun}"

if [[ ! -f "$MORPHMIMIC_SUBSET_YAML" ]]; then
  echo "Build configs/data/nyx_g1_basic.yaml with morphmimic.scripts.filter_training_motions first." >&2
  exit 1
fi

exec bash "$REPO_DIR/scripts/train_nyx_g1_8gpu.sh" "$@"
