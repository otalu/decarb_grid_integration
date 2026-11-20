#!/bin/bash
# Submit "run_pipeline.py preprocess" (the stages switched on in config/config.yaml).
# Usage: bash orcd/submit_preprocess.sh <workers: 60|90> [after_job_id]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_workers "${1:-}" "bash orcd/submit_preprocess.sh <workers: 60|90> [after_job_id]"

submit_job preprocess "$CPUS" "$MEM" "$PY_TIME" "${2:-}" "
setup_python
export PIPELINE_WORKERS=$1
python src/run_pipeline.py preprocess"
