#!/bin/bash
# Submit "run_pipeline.py decarb-inputs" (creates out/studies/study_N).
# Usage: bash orcd/submit_decarb_inputs.sh "study description" [after_job_id]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ -z "${1:-}" ]; then
  echo "Usage: bash orcd/submit_decarb_inputs.sh \"study description\" [after_job_id]" >&2
  exit 1
fi
desc_quoted=$(printf '%q' "$1")

submit_job decarb_inputs "$PY_TIME" "${2:-}" "
setup_python
export PIPELINE_WORKERS=$WORKERS
python src/run_pipeline.py decarb-inputs --description $desc_quoted"
