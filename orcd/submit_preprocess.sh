#!/bin/bash
# Submit "run_pipeline.py preprocess" (the stages switched on in config/config.yaml).
# Usage: bash orcd/submit_preprocess.sh [after_job_id]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

submit_job preprocess "$PY_CPUS" "$PY_MEM" "$PY_TIME" "${1:-}" '
setup_python
python src/run_pipeline.py preprocess'
