# ORCD (MIT Engaging) settings, read by the submit scripts in orcd/.
# Shell syntax: it is loaded by bash, both on the login node and inside each job.

# --- SLURM ---
PARTITION="mit_preemptable"
MAIL_USER=""                          # e.g. you@mit.edu (empty = no mail)
WORKERS=90                            # 60 or 90, for every job (60 -> 1000G, 90 -> 1500G; CPUs = WORKERS + 1).
                                      # The Python steps (downloads, thermal fit, decarb-inputs, OpenDSS)
                                      # and DECARB all use this many parallel workers.
DECARB_TIME="12:00:00"                # time limit of the DECARB job
PY_TIME="12:00:00"                    # time limit of the Python jobs (preprocess, decarb-inputs, network)

# --- DECARB ---
DECARB_DIR="$HOME/decarb-main"        # DECARB repository (folder with Project.toml)
SOLVER="gurobi"                       # "gurobi" or "highs" (the Gurobi license allows at most 100 workers)

# --- Software: your own setup lines ---
setup_python() {                      # activate the environment that has requirements.txt installed
  module load miniforge
  eval "$(conda shell.bash hook)"     # makes "conda activate" work in a batch job
  conda activate decarb_grid
}

setup_julia() {                       # put julia on PATH (or: module load julia/1.12.6)
  export PATH="$HOME/.juliaup/bin:$PATH"
}

setup_gurobi() {                      # only used when SOLVER="gurobi"
  export GUROBI_HOME="$HOME/gurobi1301/linux64"
  export PATH="$GUROBI_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$GUROBI_HOME/lib:${LD_LIBRARY_PATH:-}"
  export GRB_LICENSE_FILE="$HOME/gurobi.lic"
}
