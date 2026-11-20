# Shared helper for the orcd/ submit scripts (sourced, not run).
# Finds the repo from this file's location, so the repo can be cloned anywhere.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/orcd/env.sh"
SLURM_LOG_DIR="$REPO/out/logs/slurm"
mkdir -p "$SLURM_LOG_DIR"

# Workers -> memory for the DECARB job (CPUs = workers + 1 for the main process)
declare -A DECARB_MEM=( [60]="1000G" [90]="1500G" )

# submit_job NAME CPUS MEM TIME DEPENDENCY_JOB_ID BODY
#   Submits BODY as a batch job running in the repo with env.sh loaded.
#   DEPENDENCY_JOB_ID may be empty; otherwise the job waits for that job to succeed.
#   Prints the job ID.
submit_job() {
  local name="$1" cpus="$2" mem="$3" time="$4" dep="$5" body="$6"
  local opts=(--parsable --job-name="$name"
              --output="$SLURM_LOG_DIR/${name}_%j.out" --error="$SLURM_LOG_DIR/${name}_%j.err"
              --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$mem"
              --time="$time" --partition="$PARTITION")
  [ -n "$MAIL_USER" ] && opts+=(--mail-type=END,FAIL --mail-user="$MAIL_USER")
  [ -n "$dep" ] && opts+=(--dependency=afterok:"$dep")
  sbatch "${opts[@]}" <<EOF
#!/bin/bash
set -eo pipefail
source "$REPO/orcd/env.sh"
cd "$REPO"
$body
EOF
}

# Prints the decarb/ folder of study N, or of the latest study if N is empty.
# Used inside jobs, so "latest" is resolved when the job starts.
study_decarb_dir_cmd() {
  if [ -n "$1" ]; then
    echo "STUDY_DIR=\"$REPO/out/studies/study_$1/decarb\""
  else
    echo "STUDY_DIR=\"\$(ls -dv \"$REPO\"/out/studies/study_* | tail -1)/decarb\""
  fi
}
