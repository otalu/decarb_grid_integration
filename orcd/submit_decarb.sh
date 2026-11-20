#!/bin/bash
# Submit the DECARB run (src/run_decarb_study.jl) for one study.
# Usage: bash orcd/submit_decarb.sh [study_number] [after_job_id]
#   No study number: the latest study when the job starts.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ "$SOLVER" != "gurobi" ] && [ "$SOLVER" != "highs" ]; then
  echo "SOLVER in config/orcd.sh must be \"gurobi\" or \"highs\", got \"$SOLVER\"" >&2
  exit 1
fi
if [ "$SOLVER" = "gurobi" ] && [ "$WORKERS" -gt 100 ]; then
  echo "The Gurobi license allows at most 100 workers (WORKERS=$WORKERS in config/orcd.sh)" >&2
  exit 1
fi

gurobi_line=""
[ "$SOLVER" = "gurobi" ] && gurobi_line="setup_gurobi"

submit_job decarb "$DECARB_TIME" "${2:-}" "
setup_julia
$gurobi_line
$(study_decarb_dir_cmd "${1:-}")
export STUDY_DIR DECARB_DIR SOLVER
export N_WORKERS=$WORKERS
echo \"STUDY_DIR=\$STUDY_DIR  workers=\$N_WORKERS  solver=\$SOLVER\"
julia --project=\"\$DECARB_DIR\" src/run_decarb_study.jl"
