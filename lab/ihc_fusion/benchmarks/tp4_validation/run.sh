#!/usr/bin/env bash
set -euo pipefail
HARNESS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HY4_TP8_RUN=${HY4_TP8_RUN:-${HY4_TP4_RUN:?Set HY4_TP8_RUN}}
export HY4_TP8_CHECKPOINT=${HY4_TP8_CHECKPOINT:-${HY4_TP4_CHECKPOINT:-}}
ENTRY=$(basename -- "${1:?Use validate.py, benchmark.py or analyze_perf.py}")
shift
if [[ "$ENTRY" == analyze_perf.py ]]; then
  REPO=$(cd -- "$HARNESS/../../../.." && pwd)
  SAVED=$HY4_TP8_RUN
  [[ ! -d "$SAVED/performance/raw" ]] || SAVED="$SAVED/performance"
  exec "${HY4_TP8_PYTHON:-$REPO/.venv/bin/python}" -S "$HARNESS/analyze_perf.py" "$SAVED" "$@"
fi
exec bash "$HARNESS/../tp8_validation/run.sh" "$ENTRY" "$@"
