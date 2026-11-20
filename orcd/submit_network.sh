#!/bin/bash
# Submit network-inputs, opendss and results for one study, in one job.
# Usage: bash orcd/submit_network.sh <workers: 60|90> [study_number] [after_job_id]
#   No study number: the latest study when the job starts.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_workers "${1:-}" "bash orcd/submit_network.sh <workers: 60|90> [study_number] [after_job_id]"

study_arg=""
[ -n "${2:-}" ] && study_arg="--study $2"

submit_job network "$CPUS" "$MEM" "$PY_TIME" "${3:-}" "
setup_python
export PIPELINE_WORKERS=$1
python src/run_pipeline.py network-inputs $study_arg
python src/run_pipeline.py opendss $study_arg
python src/run_pipeline.py results $study_arg"
