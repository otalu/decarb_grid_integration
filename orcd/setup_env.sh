#!/bin/bash
# One-time setup of the pipeline's Python environment on ORCD.
# Creates the conda environment CONDA_ENV (config/orcd.sh) with ORCD's miniforge
# module and installs requirements.txt. If it already exists, only updates the
# packages. Ends with an import check.
# Run on the login node: bash orcd/setup_env.sh
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/config/orcd.sh"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "Run this on the login node with: bash orcd/setup_env.sh (not with sbatch)" >&2
  exit 1
fi
if [ -z "${CONDA_ENV:-}" ]; then
  echo "CONDA_ENV is not set in config/orcd.sh" >&2
  exit 1
fi

module load miniforge
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "Conda environment '$CONDA_ENV' exists; updating its packages from requirements.txt"
else
  echo "Creating conda environment '$CONDA_ENV' (Python 3.12)"
  conda create -y -n "$CONDA_ENV" python=3.12
fi

conda activate "$CONDA_ENV"
pip install -r "$REPO/requirements.txt"

echo "Checking imports..."
python - <<'EOF'
import importlib, sys
failed = []
for name in ["pandas", "numpy", "pyarrow", "scipy", "sklearn", "matplotlib",
             "geopandas", "shapely", "boto3", "requests", "pvlib", "yaml",
             "opendssdirect"]:
    try:
        importlib.import_module(name)
    except Exception as e:
        failed.append(f"{name}: {e}")
if failed:
    print("FAILED imports:\n  " + "\n  ".join(failed))
    sys.exit(1)
import pandas
print(f"ok: Python {sys.version.split()[0]}, pandas {pandas.__version__}")
EOF
echo "Done. The jobs activate '$CONDA_ENV' through setup_python in config/orcd.sh."
