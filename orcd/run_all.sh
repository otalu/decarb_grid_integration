#!/bin/bash
# Submit the whole chain; each job starts only if the previous one succeeded:
#   preprocess -> decarb-inputs -> DECARB -> network-inputs + opendss + results
# Every job gets the same worker count (60 -> 1000G, 90 -> 1500G).
# The DECARB and network jobs use the study that decarb-inputs creates (the latest).
# Usage: bash orcd/run_all.sh <workers: 60|90> "study description"
set -e
DIR="$(dirname "${BASH_SOURCE[0]}")"

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
  echo "Usage: bash orcd/run_all.sh <workers: 60|90> \"study description\"" >&2
  exit 1
fi

j1=$(bash "$DIR/submit_preprocess.sh" "$1")
j2=$(bash "$DIR/submit_decarb_inputs.sh" "$1" "$2" "$j1")
j3=$(bash "$DIR/submit_decarb.sh" "$1" "" "$j2")
j4=$(bash "$DIR/submit_network.sh" "$1" "" "$j3")
echo "Submitted: preprocess=$j1  decarb_inputs=$j2  decarb=$j3  network=$j4"
echo "Job logs: $(cd "$DIR/.." && pwd)/out/logs/slurm/"
