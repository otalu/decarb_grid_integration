#!/bin/bash
# Submit network-inputs, opendss and results for one study, in one job.
# Usage: bash orcd/submit_network.sh [study_number] [after_job_id]
#   No study number: the latest study when the job starts.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

study_arg=""
[ -n "${1:-}" ] && study_arg="--study $1"

submit_job network "$PY_TIME" "${2:-}" "
setup_python
export PIPELINE_WORKERS=$WORKERS
python src/run_pipeline.py network-inputs $study_arg
python src/run_pipeline.py opendss $study_arg
python src/run_pipeline.py results $study_arg"
