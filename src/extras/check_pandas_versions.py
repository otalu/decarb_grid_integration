"""
Check that decarb-inputs writes the same files under two pandas versions.

Builds the DECARB input files for a few buildings in every case with the
pipeline's own per-building code (create_decarb_inputs.process_building), into
out/pandas_check/pandas_<version>/, then compares two such folders file by file.
Elevation is fixed at 0 m (no web lookups); everything else is as in a real run.

Usage (run once in each Python environment, then compare):
    python src/extras/check_pandas_versions.py generate [--n 5] [--ids 104 1328]
    python src/extras/check_pandas_versions.py compare out/pandas_check/pandas_2.3.3 out/pandas_check/pandas_3.0.6
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from config.loader import load_config, load_decarb_config, resolve_timestep  # noqa: E402
import create_decarb_inputs as cdi  # noqa: E402
from fit_thermal_model import thermal_coeff_csv_path  # noqa: E402


def generate(n, ids):
    cfg = load_config()
    dcfg = load_decarb_config()
    dcfg = dict(dcfg, timestep=resolve_timestep(cfg, dcfg))
    input_dir, output_dir, state = cfg["paths"]["input_dir"], cfg["paths"]["output_dir"], cfg["state"]

    coeffs = pd.read_csv(thermal_coeff_csv_path(cfg))
    regression_coeffs = {str(int(r.bldg_id)): {"k1": r.k1, "k2": r.k2, "k3": r.k3}
                         for r in coeffs.itertuples()}
    if ids:
        building_ids = [int(b) for b in ids]
    else:
        matched = pd.read_parquet(input_dir / "feeder_summaries_matched.parquet")
        col = "nrel_climate_match" if "nrel_climate_match" in matched.columns else "nrel_county_match"
        pool = sorted(b for b in matched[col].dropna().astype(int).unique()
                      if str(b) in regression_coeffs)
        building_ids = [int(pool[i]) for i in np.linspace(0, len(pool) - 1, min(n, len(pool))).astype(int)]

    metadata_df = pd.read_parquet(input_dir / f"{state}_upgrade0.parquet")
    hp_summary_df = pd.read_csv(output_dir / "heatpump_files" / state / "building_hp_summary.csv")
    templates = cdi.load_templates(dcfg["template_dir"], dcfg["outputs"])
    id_col = "bldg_id" if "bldg_id" in metadata_df.columns else "building_id"
    sub = metadata_df[metadata_df[id_col].isin(building_ids)]
    elevation_cache = {(float(a), float(b)): 0.0 for a, b in
                       sub[["in.weather_file_latitude", "in.weather_file_longitude"]].itertuples(index=False)}

    out = output_dir / "pandas_check" / f"pandas_{pd.__version__}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cdi._worker_init(cfg, dcfg, metadata_df, regression_coeffs, templates,
                     elevation_cache, hp_summary_df, out)

    cases = []
    for elec in dcfg["electrification"]["levels"]:
        for tariff in dcfg["tariffs"]:
            cases.append((tariff, elec, False, f"{tariff['name']}_heat_{elec}_solar_0"))
            if dcfg.get("rooftop_solar", {}).get("enabled", False):
                cases.append((tariff, elec, True, f"{tariff['name']}_heat_{elec}_solar_100"))

    failures = 0
    for tariff, elec, solar, case in cases:
        for b in building_ids:
            r = cdi.process_building((b, tariff, elec, solar, case))
            if not r["success"]:
                failures += 1
                print(f"FAILED {case}/{b}: {r['error']}")
    print(f"pandas {pd.__version__}: {len(building_ids)} buildings x {len(cases)} cases "
          f"-> {out} ({failures} failures)")
    print("buildings:", building_ids)


def _compare_csv(a, b):
    """Return a list of differences between two CSV files (empty if equal)."""
    da = pd.read_csv(a, header=None, dtype=str, keep_default_na=False)
    db = pd.read_csv(b, header=None, dtype=str, keep_default_na=False)
    if da.shape != db.shape:
        return [f"shape {da.shape} vs {db.shape}"]
    diffs = []
    for (i, j), va in np.ndenumerate(da.values):
        vb = db.values[i, j]
        if va == vb:
            continue
        try:
            fa, fb = float(va), float(vb)
            if np.isclose(fa, fb, rtol=1e-9, atol=1e-12):
                continue
        except ValueError:
            pass
        diffs.append(f"row {i} col {j}: {va!r} vs {vb!r}")
    return diffs


def compare(dir_a, dir_b):
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    files_a = {p.relative_to(dir_a) for p in dir_a.rglob("*.csv")}
    files_b = {p.relative_to(dir_b) for p in dir_b.rglob("*.csv")}
    only = sorted(files_a ^ files_b)
    for f in only:
        print(f"only in {'A' if f in files_a else 'B'}: {f}")
    n_diff = 0
    for f in sorted(files_a & files_b):
        diffs = _compare_csv(dir_a / f, dir_b / f)
        if diffs:
            n_diff += 1
            print(f"DIFFERENT {f}: {len(diffs)} values, e.g. {diffs[:3]}")
    print(f"\n{len(files_a & files_b)} files compared, {n_diff} different, "
          f"{len(only)} missing on one side")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("generate")
    g.add_argument("--n", type=int, default=5, help="number of matched buildings (spread over the ID range)")
    g.add_argument("--ids", nargs="*", help="specific building IDs instead of --n")
    c = sp.add_parser("compare")
    c.add_argument("dir_a")
    c.add_argument("dir_b")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args.n, args.ids)
    else:
        compare(args.dir_a, args.dir_b)
