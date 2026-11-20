"""
OpenDSS Study Runner (Python, multiprocessing)
───────────────────────────────────────────────────────────────────────────────
Runs time-series power flow for each DECARB case in a study.

Each case in the study's opendss/{case}/ directory is self-contained (written
by network-inputs). The step size and number of steps come from the
'Solve mode=yearly stepsize=... number=...' line of each feeder's Master.dss,
so hourly and 15-min cases (and date ranges) run without changes here.

Usage:
    python src/run_opendss.py [--study N] [--workers 20]
───────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from config.loader import load_config, job_workers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[1]
SOLVE_LINE = re.compile(
    r"^\s*solve\s+mode\s*=\s*yearly\s+stepsize\s*=\s*(\d+)\s*([hm])\s+number\s*=\s*(\d+)",
    re.IGNORECASE | re.MULTILINE)


def read_solve_settings(master_path):
    """(stepsize string for OpenDSS, step length in hours, number of steps)
    from the case's 'Solve mode=yearly stepsize=... number=...' line."""
    m = SOLVE_LINE.search(Path(master_path).read_text())
    if not m:
        raise ValueError(f"No 'Solve mode=yearly stepsize=... number=...' line in {master_path}; "
                         f"re-run network-inputs for this study")
    size, unit, number = int(m.group(1)), m.group(2).lower(), int(m.group(3))
    step_hours = size if unit == "h" else size / 60
    return f"{size}{unit}", step_hours, number


# ── Discovery ─────────────────────────────────────────────────────────────

def find_study_dir(output_dir, study_num=None):
    studies_root = Path(output_dir) / "studies"
    if study_num is not None:
        d = studies_root / f"study_{study_num}"
        if not d.exists():
            raise FileNotFoundError(f"Study not found: {d}")
        return d
    dirs = sorted(
        [d for d in studies_root.iterdir() if d.is_dir() and d.name.startswith("study_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    if not dirs:
        raise FileNotFoundError("No studies found")
    return dirs[-1]


def discover_feeders(zone_template_dir):
    """Find all feeders with Master.dss in the template."""
    feeders = []
    zone_dir = Path(zone_template_dir)
    for sub in sorted(zone_dir.iterdir()):
        if not sub.is_dir():
            continue
        # Skip non-substation directories (e.g. opendss_loadshape_files)
        if not sub.name.startswith("p"):
            continue
        for feeder_dir in sorted(sub.iterdir()):
            if not feeder_dir.is_dir():
                continue
            master = feeder_dir / "Master.dss"
            if master.exists():
                feeders.append({
                    "substation": sub.name,
                    "feeder": feeder_dir.name,
                    "master": str(master),
                    "dir": str(feeder_dir),
                })
    return feeders


# ── Worker function (runs in subprocess) ──────────────────────────────────

def simulate_feeder(task):
    """Run the time-series power flow for a single feeder in a single case.

    task: dict with keys: case, substation, feeder, master, feeder_dir
    The feeder_dir already contains rewritten LoadShapes.dss and Loads.dss
    pointing to our case-specific kW/kvar profiles.
    Returns: dict with status, timing, and result data
    """
    import opendssdirect as dss
    import os

    case_name = task["case"]
    feeder_name = task["feeder"]
    substation = task["substation"]
    master_path = task["master"]
    feeder_dir = task["feeder_dir"]

    t0 = time.time()

    try:
        stepsize, step_hours, num_steps = read_solve_settings(master_path)
        dss.Text.Command("Clear")
        os.chdir(feeder_dir)

        with open(master_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            s = line.strip()
            if not s:
                continue
            low = s.lower()
            if low.startswith("solve") or low.startswith("export") or low.startswith("plot "):
                continue
            dss.Text.Command(s)

        # Set mode and run
        dss.Text.Command(f"Set mode=yearly stepsize={stepsize} number=1")

        hours = np.empty(num_steps)
        v_min = np.empty(num_steps)
        v_max = np.empty(num_steps)
        v_mean = np.empty(num_steps)
        tot_kw = np.empty(num_steps)
        tot_kvar = np.empty(num_steps)
        loading = np.empty(num_steps)

        # Get head line for loading measurement
        dss.Lines.First()
        head_line = dss.Lines.Name()
        all_lines = [head_line]
        while dss.Lines.Next() != 0:
            all_lines.append(dss.Lines.Name())
        # Prefer the head-of-feeder line: the one touching the substation,
        # e.g. "p100uhs" for substation p100uhs0_1247
        hs_prefix = substation.lower().split("hs")[0] + "hs"
        hs_lines = [l for l in all_lines if hs_prefix in l.lower()]
        if hs_lines:
            head_line = hs_lines[0]

        for step in range(num_steps):
            dss.Solution.Solve()
            hours[step] = step * step_hours

            # Bus voltages
            all_vmag = dss.Circuit.AllBusMagPu()
            valid = [v for v in all_vmag if v > 0.01]
            if valid:
                v_min[step] = min(valid)
                v_max[step] = max(valid)
                v_mean[step] = sum(valid) / len(valid)
            else:
                v_min[step] = v_max[step] = v_mean[step] = float("nan")

            # Total power
            p = dss.Circuit.TotalPower()
            tot_kw[step] = -p[0]
            tot_kvar[step] = -p[1]

            # Head line loading
            dss.Circuit.SetActiveElement(f"Line.{head_line}")
            currents = dss.CktElement.CurrentsMagAng()
            n_cond = dss.CktElement.NumConductors()
            i_max = max(currents[2 * i] for i in range(n_cond))  # mag values at even indices
            norm_amps = dss.CktElement.NormalAmps()
            loading[step] = (i_max / norm_amps * 100) if norm_amps > 0 else 0.0

        elapsed = round(time.time() - t0, 2)

        result_df = pd.DataFrame({
            "feeder": feeder_name,
            "substation": substation,
            "hour": hours.astype(int) if step_hours == 1 else hours,   # hours since start
            "v_min_pu": v_min,
            "v_max_pu": v_max,
            "v_mean_pu": v_mean,
            "total_kW": tot_kw,
            "total_kvar": tot_kvar,
            "head_loading_pct": loading,
        })

        return {
            "case": case_name,
            "feeder": feeder_name,
            "substation": substation,
            "status": "ok",
            "elapsed_s": elapsed,
            "data": result_df,
            "error": "",
        }

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        return {
            "case": case_name,
            "feeder": feeder_name,
            "substation": substation,
            "status": "error",
            "elapsed_s": elapsed,
            "data": None,
            "error": str(e),
        }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenDSS Study Runner")
    parser.add_argument("--study", type=int, default=None, help="Study number (default: latest)")
    parser.add_argument("--workers", type=int, default=job_workers(20),
                        help="Number of parallel workers (default: 20, or the orcd/ job's worker count)")
    args = parser.parse_args()

    # When started by run_pipeline.py, also write to that run's log file
    from run_log import attach_run_log
    attach_run_log()

    cfg = load_config()
    output_dir = cfg["paths"]["output_dir"]
    study_dir = find_study_dir(output_dir, args.study)
    log.info(f"Study: {study_dir}")

    # Discover cases — each case dir has its own copy of feeder files
    # with rewritten LoadShapes.dss and Loads.dss
    opendss_root = study_dir / "opendss"
    if not opendss_root.exists():
        log.error(f"No opendss directory in study: {opendss_root}")
        return
    cases = sorted([d.name for d in opendss_root.iterdir() if d.is_dir()])
    log.info(f"Found {len(cases)} cases")

    # Results directory
    results_root = study_dir / "opendss_results"
    results_root.mkdir(exist_ok=True)

    # Build task list — skip completed cases
    all_tasks = []
    for case_name in cases:
        case_results_dir = results_root / case_name
        zone_csv = case_results_dir / "zone_yearly.csv"
        if zone_csv.exists():
            log.info(f"  {case_name}: already complete — skipping")
            continue

        case_results_dir.mkdir(exist_ok=True)
        case_opendss_dir = opendss_root / case_name

        # Discover feeders from this case's own directory
        feeders = discover_feeders(case_opendss_dir)
        if not feeders:
            log.warning(f"  {case_name}: no feeders found")
            continue

        for feeder in feeders:
            all_tasks.append({
                "case": case_name,
                "substation": feeder["substation"],
                "feeder": feeder["feeder"],
                "master": feeder["master"],
                "feeder_dir": feeder["dir"],
            })

    if not all_tasks:
        log.info("Nothing to run — all cases complete.")
        return

    n_workers = min(args.workers, len(all_tasks))
    log.info(f"Running {len(all_tasks)} feeder-case tasks with {n_workers} workers")

    # Count expected feeders per case for incremental saving
    feeders_per_case = {}
    for task in all_tasks:
        feeders_per_case[task["case"]] = feeders_per_case.get(task["case"], 0) + 1

    # Run
    t_start = time.time()
    completed = 0
    errors = 0
    case_results = {}   # case_name -> list of DataFrames
    case_errors = {}    # case_name -> error count

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(simulate_feeder, task): task for task in all_tasks}

        for future in as_completed(futures):
            result = future.result()
            completed += 1
            case_name = result["case"]

            if case_name not in case_results:
                case_results[case_name] = []
                case_errors[case_name] = 0

            if result["status"] == "ok":
                icon = "✓"
                case_results[case_name].append(result["data"])
            else:
                icon = "✗"
                errors += 1
                case_errors[case_name] += 1
                err_file = results_root / case_name / "errors.log"
                with open(str(err_file), "a") as f:
                    f.write(f"{result['substation']}/{result['feeder']}: {result['error']}\n")

            elapsed = time.time() - t_start
            eta = elapsed / completed * (len(all_tasks) - completed) if completed > 0 else 0
            log.info(f"  {icon} [{completed}/{len(all_tasks)}] "
                     f"{case_name}/{result['feeder']} "
                     f"({result['elapsed_s']}s) | "
                     f"ETA: {eta/60:.1f}min")

            # Write case results as soon as all its feeders are done
            n_done = len(case_results[case_name]) + case_errors[case_name]
            if n_done == feeders_per_case[case_name] and case_results[case_name]:
                zone_df = pd.concat(case_results[case_name], ignore_index=True)
                out_path = results_root / case_name / "zone_yearly.csv"
                zone_df.to_csv(str(out_path), index=False)
                log.info(f"  Wrote {out_path} ({len(case_results[case_name])} feeders)")
                del case_results[case_name]  # free memory

    total_time = time.time() - t_start
    log.info(f"Done: {completed - errors} ok, {errors} errors, {total_time/60:.1f} min total")


if __name__ == "__main__":
    main()
