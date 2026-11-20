# Shared helper for the orcd/ submit scripts (sourced, not run).
# Finds the repo from this file's location, so the repo can be cloned anywhere.
# All settings are in config/orcd.sh.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTINGS="$REPO/config/orcd.sh"
source "$SETTINGS"
SLURM_LOG_DIR="$REPO/out/logs/slurm"
mkdir -p "$SLURM_LOG_DIR"

# Workers -> memory, for every job (CPUs = workers + 1 for the main process)
declare -A JOB_MEM=( [60]="1000G" [90]="1500G" )
MEM="${JOB_MEM[${WORKERS:-x}]:-}"
if [ -z "$MEM" ]; then
  echo "WORKERS in config/orcd.sh must be 60 or 90 (memory: 60 -> 1000G, 90 -> 1500G), got \"${WORKERS:-}\"" >&2
  exit 1
fi
CPUS=$(( WORKERS + 1 ))

# submit_job NAME TIME DEPENDENCY_JOB_ID BODY
#   Submits BODY as a batch job with CPUS and MEM, running in the repo with
#   config/orcd.sh loaded. DEPENDENCY_JOB_ID may be empty; otherwise the job
#   waits for that job to succeed. Prints the job ID.
submit_job() {
  local name="$1" time="$2" dep="$3" body="$4"
  local opts=(--parsable --job-name="$name"
              --output="$SLURM_LOG_DIR/${name}_%j.out" --error="$SLURM_LOG_DIR/${name}_%j.err"
              --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM"
              --time="$time" --partition="$PARTITION")
  [ -n "$MAIL_USER" ] && opts+=(--mail-type=END,FAIL --mail-user="$MAIL_USER")
  [ -n "$dep" ] && opts+=(--dependency=afterok:"$dep")
  sbatch "${opts[@]}" <<EOF
#!/bin/bash
set -eo pipefail
source "$SETTINGS"
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
