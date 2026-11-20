#!/bin/bash
# Submit the DECARB run (src/run_decarb_study.jl) for one study.
# Usage: bash orcd/submit_decarb.sh <workers: 60|90> [study_number] [after_job_id]
#   No study number: the latest study when the job starts.
#   Memory follows the worker count (60 -> 1000G, 90 -> 1500G); CPUs = workers + 1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

workers="${1:-}"
mem="${DECARB_MEM[$workers]:-}"
if [ -z "$mem" ]; then
  echo "Usage: bash orcd/submit_decarb.sh <workers> [study_number] [after_job_id]" >&2
  echo "workers must be 60 or 90 (memory: 60 -> 1000G, 90 -> 1500G)" >&2
  exit 1
fi
if [ "$SOLVER" = "gurobi" ] && [ "$workers" -gt 100 ]; then
  echo "The Gurobi license allows at most 100 workers (got $workers)" >&2
  exit 1
fi
if [ "$SOLVER" != "gurobi" ] && [ "$SOLVER" != "highs" ]; then
  echo "SOLVER in orcd/env.sh must be \"gurobi\" or \"highs\", got \"$SOLVER\"" >&2
  exit 1
fi

gurobi_line=""
[ "$SOLVER" = "gurobi" ] && gurobi_line="setup_gurobi"

submit_job decarb "$((workers + 1))" "$mem" "$DECARB_TIME" "${3:-}" "
setup_julia
$gurobi_line
$(study_decarb_dir_cmd "${2:-}")
export STUDY_DIR DECARB_DIR SOLVER
export N_WORKERS=$workers
echo \"STUDY_DIR=\$STUDY_DIR  workers=\$N_WORKERS  solver=\$SOLVER\"
julia --project=\"\$DECARB_DIR\" src/run_decarb_study.jl"
