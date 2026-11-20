#!/bin/bash
# Submit "run_pipeline.py decarb-inputs" (creates out/studies/study_N).
# Usage: bash orcd/submit_decarb_inputs.sh <workers: 60|90> "study description" [after_job_id]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
usage="bash orcd/submit_decarb_inputs.sh <workers: 60|90> \"study description\" [after_job_id]"
check_workers "${1:-}" "$usage"
if [ -z "${2:-}" ]; then
  echo "Usage: $usage" >&2
  exit 1
fi
desc_quoted=$(printf '%q' "$2")

submit_job decarb_inputs "$CPUS" "$MEM" "$PY_TIME" "${3:-}" "
setup_python
export PIPELINE_WORKERS=$1
python src/run_pipeline.py decarb-inputs --description $desc_quoted"
