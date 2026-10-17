#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:?Usage: bash scripts/run.sh fixed|progressive|smoke|evaluate [arguments]}"
shift
case "$mode" in
  fixed) entry=train_thuman_cycle_4views.py ;;
  progressive) entry=train_thuman_cycle_progressive.py ;;
  smoke) entry=test_cycle_progressive.py ;;
  evaluate) entry=evaluate.py ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ -n "${SPINS_CONTAINER:-}" ]]; then
  exec "${SPINS_CONTAINER_RUNTIME:-singularity}" exec --nv "$SPINS_CONTAINER" \
    "${SPINS_PYTHON:-python}" -u "$entry" "$@"
fi
exec "${SPINS_PYTHON:-python}" -u "$entry" "$@"
