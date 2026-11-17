"""
Network analysis input generation.

Reads DECARB outputs from a study directory and produces:
  - GenX demand profiles (scope=statewide): aggregates building-level
    electricity purchases into bus-level Load_MW columns
  - OpenDSS loadshapes (scope=zone): scales or replaces feeder-level
    loadshapes with DECARB output profiles

Each DECARB case within a study can produce multiple GenX or OpenDSS
cases with different network parameters.
"""

import logging
import os
import re
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[1]))
from config.loader import (load_config, load_decarb_config, load_network_config,
                           resolve_hour_indices, resolve_step_indices, resolve_timestep)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet reader
# ---------------------------------------------------------------------------

def read_parquet(path):
    try:
        return pd.read_parquet(str(path), engine="fastparquet")
    except Exception:
        return pd.read_parquet(str(path), engine="pyarrow")


# ---------------------------------------------------------------------------
# Building-to-bus mapping
# ---------------------------------------------------------------------------

def load_building_map(input_dir=None, output_dir=None, study_dir=None):
    """Load the bus-to-building matching results.

    Search order:
      1. Study-level matching file (study_dir/feeder_summaries_matched.parquet)
      2. Explicit building map files in output_dir
      3. Global matching file (input_dir/feeder_summaries_matched.parquet)

    Returns a DataFrame with columns: bldg_id, count, and optionally
    substation/zone info for multi-bus aggregation.
    """
    # 1. Study-level matching (preferred — ties results to the study)
    if study_dir is not None:
        study_matched = Path(study_dir) / "feeder_summaries_matched.parquet"
        if study_matched.exists():
            log.info(f"Using study-level matching: {study_matched}")
            return _parse_feeder_matched(read_parquet(study_matched))

    # 2. Check explicit building map files in output_dir
    if output_dir is not None:
        for path in (
            output_dir / "building_load_matching" / "building_map.parquet",
            output_dir / "building_load_matching" / "building_map.csv",
            output_dir / "ercot_substation_nrel_map.parquet",
            output_dir / "ercot_substation_nrel_map.csv",
        ):
            if path.exists():
                df = read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(str(path))
                if "bldg_id" in df.columns and "count" in df.columns:
                    return df

    # 3. Global matching file (fallback)
    if input_dir is not None:
        matched_path = input_dir / "feeder_summaries_matched.parquet"
        if matched_path.exists():
            log.info(f"Using global matching: {matched_path}")
            return _parse_feeder_matched(read_parquet(matched_path))

    raise FileNotFoundError(
        "No building map found. Run building_load_matching first, "
        "or copy a matching file into the study directory.")


def _parse_feeder_matched(feeder_df):
    """Convert feeder_summaries_matched parquet into a building map DataFrame."""
    bldg_col = None
    for col in ("nrel_climate_match", "nrel_county_match"):
        if col in feeder_df.columns:
            bldg_col = col
            break

    if bldg_col is None:
        raise ValueError("Matching file has no nrel_climate_match or nrel_county_match column")

    matched = feeder_df[feeder_df[bldg_col].notna()].copy()
    matched["bldg_id"] = matched[bldg_col].astype(int)
    sub_col = "substation" if "substation" in matched.columns else None

    if sub_col:
        grouped = (matched.groupby([sub_col, "bldg_id"])
                   .size().reset_index(name="count"))
        grouped = grouped.rename(columns={sub_col: "substation"})
    else:
        grouped = (matched.groupby("bldg_id")
                   .size().reset_index(name="count"))

    if "Name" in matched.columns:
        load_ids = (matched.groupby("bldg_id")["Name"]
                    .first().reset_index()
                    .rename(columns={"Name": "load_id"}))
        grouped = grouped.merge(load_ids, on="bldg_id", how="left")

    log.info(f"Building map: {len(grouped)} entries, "
             f"{grouped['bldg_id'].nunique()} unique buildings")
    return grouped


def build_substation_lookup(building_map_df):
    """Build substation → {bldg_id: count} lookup from the building map.

    The building map has one row per bus. Each bus maps to a building
    with a count (how many times that building appears at that bus).
    Group by substation to get per-substation building counts.
    """
    sub_col = None
    for col in ("substation", "sub_id", "network_bus_id"):
        if col in building_map_df.columns:
            sub_col = col
            break

    if sub_col is None:
        return None

    lookup = {}
    for sub_id, group in building_map_df.groupby(sub_col):
        counts = {}
        for _, row in group.iterrows():
            bldg_id = str(int(row["bldg_id"]))
            c = int(row.get("count", 1))
            counts[bldg_id] = counts.get(bldg_id, 0) + c
        lookup[str(sub_id)] = counts

    return lookup


# ---------------------------------------------------------------------------
# DECARB output reading
# ---------------------------------------------------------------------------

def read_decarb_ts(decarb_case_dir, bldg_id):
    """Read DECARB optimizer output timeseries for a building.

    The DECARB optimizer writes ts.csv with columns like:
    buy, sell, hvac_HT, hvac_AC, pv, etc., all average kW per period
    (vQbuy / vQsell are powers in DECARB's electricity balance).
    """
    ts_path = decarb_case_dir / str(bldg_id) / "out" / "ts.csv"
    if not ts_path.exists():
        return None
    try:
        df = pd.read_csv(str(ts_path))
        if df.empty:
            return None
        return df
    except Exception:
        return None


def read_decarb_tm(decarb_case_dir, bldg_id):
    """Read the DECARB input tm.csv to get consumption data when
    DECARB outputs aren't available yet (pre-optimization)."""
    tm_path = decarb_case_dir / str(bldg_id) / "in" / "tm.csv"
    if not tm_path.exists():
        return None
    return pd.read_csv(str(tm_path), index_col=0)


# ---------------------------------------------------------------------------
# Failed-building substitution
# ---------------------------------------------------------------------------

SUMMER_DEMAND_TAG = "out.qoi.electricity.maximum_daily_peak_summer..kw"
WINTER_DEMAND_TAG = "out.qoi.electricity.maximum_daily_peak_winter..kw"


def build_substitution_map(decarb_case_dir, building_ids, cfg):
    """For each failed building, find the closest successful building
    by peak demand (summer, winter) and return a substitution map.

    Uses the same matching metric as building_load_matching.py:
    Euclidean distance in (summer_peak_kw, winter_peak_kw) space.

    Returns:
        dict: {failed_bldg_id: substitute_bldg_id}
    """
    # Partition into successful and failed
    successful = []
    failed = []
    for bldg_id in building_ids:
        bldg_id = int(bldg_id)
        ts_path = decarb_case_dir / str(bldg_id) / "out" / "ts.csv"
        if ts_path.exists():
            successful.append(bldg_id)
        else:
            failed.append(bldg_id)

    if not failed:
        return {}

    if not successful:
        log.warning("No successful buildings in case — cannot substitute")
        return {}

    log.info(f"Substitution: {len(failed)} failed, {len(successful)} successful")

    # Load metadata with peak demand columns
    input_dir = cfg["paths"]["input_dir"]
    state = cfg.get("state", "TX")
    meta_path = input_dir / f"{state}_upgrade0.parquet"
    if not meta_path.exists():
        log.warning(f"Metadata not found at {meta_path} — cannot substitute")
        return {}

    meta = read_parquet(meta_path)
    if "bldg_id" not in meta.columns:
        meta = meta.reset_index().rename(columns={"index": "bldg_id"})

    needed_cols = [SUMMER_DEMAND_TAG, WINTER_DEMAND_TAG]
    if not all(c in meta.columns for c in needed_cols):
        log.warning("Metadata missing peak demand columns — cannot substitute")
        return {}

    meta = meta[["bldg_id"] + needed_cols].dropna()
    meta["bldg_id"] = meta["bldg_id"].astype(int)
    meta = meta.set_index("bldg_id")

    # Build KD-tree from successful buildings
    success_meta = meta.loc[meta.index.isin(successful)].copy()
    if success_meta.empty:
        log.warning("No successful buildings found in metadata — cannot substitute")
        return {}

    success_vectors = success_meta[needed_cols].values
    tree = cKDTree(success_vectors)

    # For each failed building, find nearest successful neighbor
    sub_map = {}
    for bldg_id in failed:
        if bldg_id not in meta.index:
            log.warning(f"Building {bldg_id} not in metadata — skipping substitution")
            continue
        target = meta.loc[bldg_id, needed_cols].values.reshape(1, -1)
        dist, idx = tree.query(target, k=1)
        substitute_id = int(success_meta.index[idx[0]])
        sub_map[bldg_id] = substitute_id
        log.debug(f"Substitute {bldg_id} -> {substitute_id} (dist={dist[0]:.2f} kW)")

    log.info(f"Substitution map: {len(sub_map)} buildings mapped")
    return sub_map


# ---------------------------------------------------------------------------
# GenX demand generation
# ---------------------------------------------------------------------------

def to_hourly(series_15min):
    """Convert 15-min series to hourly by taking the max of each 4 intervals."""
    arr = series_15min.values
    if len(arr) % 4 != 0:
        raise ValueError("15-min series length not divisible by 4")
    return arr.reshape(-1, 4).max(axis=1)


def aggregate_demand_1bus(decarb_case_dir, building_map_df, n_hours=8760,
                          sub_map=None, hour_indices=None):
    """Aggregate building-level electricity purchases into a single
    system-wide demand profile (MW) for 1-bus GenX mode.

    Uses the building map counts to scale each building's demand.
    For failed buildings, sub_map provides the substitute building ID
    whose profile is used instead.
    """
    if sub_map is None:
        sub_map = {}

    # Get unique buildings and their total counts across all substations
    bldg_counts = building_map_df.groupby("bldg_id")["count"].sum().to_dict()

    total_mw = np.zeros(n_hours)
    processed = 0
    substituted = 0
    skipped = 0

    for bldg_id, count in bldg_counts.items():
        bldg_id = int(bldg_id)
        ts_df = read_decarb_ts(decarb_case_dir, bldg_id)

        if ts_df is None or "buy" not in ts_df.columns:
            # Try substitution
            sub_id = sub_map.get(bldg_id)
            if sub_id is not None:
                ts_df = read_decarb_ts(decarb_case_dir, sub_id)
                if ts_df is not None and "buy" in ts_df.columns:
                    substituted += 1
                else:
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue

        buy_kw = ts_df["buy"].values

        # Convert to MW and scale by count
        buy_mw = buy_kw / 1000.0 * count

        # Handle 15-min vs hourly
        if len(buy_mw) == 35040:
            buy_mw = to_hourly(pd.Series(buy_mw))

        if hour_indices is not None and len(buy_mw) == 8760:
            buy_mw = buy_mw[hour_indices]

        if len(buy_mw) != n_hours:
            log.warning(f"Building {bldg_id}: unexpected length {len(buy_mw)}")
            skipped += 1
            continue

        total_mw += buy_mw
        processed += 1

    log.info(f"1-bus aggregation: {processed} buildings "
             f"({substituted} substituted), {skipped} skipped")
    return total_mw


def aggregate_demand_multibus(decarb_case_dir, substation_lookup,
                              bus_mapping_df=None, n_hours=8760,
                              sub_map=None, hour_indices=None):
    """Aggregate building-level demand per substation (or per bus zone).

    Returns a dict of {column_name: demand_array_MW}.
    For failed buildings, sub_map provides the substitute building ID.
    """
    if sub_map is None:
        sub_map = {}

    if bus_mapping_df is not None:
        sub_to_zone = dict(zip(
            bus_mapping_df.iloc[:, 0].astype(str),
            bus_mapping_df.iloc[:, 1].astype(str)))
    else:
        sub_to_zone = None

    demand_by_key = {}
    processed = 0

    for sub_id, bldg_counts in substation_lookup.items():
        sub_mw = np.zeros(n_hours)

        for bldg_id_str, count in bldg_counts.items():
            bldg_id = int(bldg_id_str)
            ts_df = read_decarb_ts(decarb_case_dir, bldg_id)

            if ts_df is None or "buy" not in ts_df.columns:
                sub_id_sub = sub_map.get(bldg_id)
                if sub_id_sub is not None:
                    ts_df = read_decarb_ts(decarb_case_dir, sub_id_sub)
                if ts_df is None or "buy" not in ts_df.columns:
                    continue

            buy_mw = ts_df["buy"].values / 1000.0 * count
            if len(buy_mw) == 35040:
                buy_mw = to_hourly(pd.Series(buy_mw))
            if hour_indices is not None and len(buy_mw) == 8760:
                buy_mw = buy_mw[hour_indices]
            if len(buy_mw) == n_hours:
                sub_mw += buy_mw

        if sub_to_zone is not None:
            zone = sub_to_zone.get(sub_id, sub_id)
            key = f"Load_MW_{zone}"
        else:
            key = f"Load_MW_{sub_id}"

        if key in demand_by_key:
            demand_by_key[key] += sub_mw
        else:
            demand_by_key[key] = sub_mw.copy()

        processed += 1

    log.info(f"Multi-bus aggregation: {processed} substations -> "
             f"{len(demand_by_key)} zones")
    return demand_by_key


def write_genx_demand(demand_data, output_dir, ncfg):
    """Write GenX-format Demand_data.csv.

    demand_data: either a 1D array (1-bus) or dict of {col: array} (multi-bus).
    """
    genx_cfg = ncfg["genx"]

    system_dir = output_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(demand_data, np.ndarray):
        demand_cols = {"Load_MW_z1": demand_data}
    else:
        demand_cols = demand_data

    first_arr = next(iter(demand_cols.values()))
    n_hours = len(first_arr)

    rows = []
    for i in range(n_hours):
        row = {"Time_Index": i + 1}
        for col_name, arr in demand_cols.items():
            row[col_name] = round(float(arr[i]), 5)

        if i == 0:
            row.update({
                "Voll": genx_cfg.get("voll", 2000000),
                "Demand_Segment": genx_cfg.get("demand_segment", 1),
                "Cost_of_Demand_Curtailment_per_MW": genx_cfg.get("cost_curtailment", 1),
                "Max_Demand_Curtailment": genx_cfg.get("max_curtailment", 1),
                "$/MWh": genx_cfg.get("cost_mwh", 2000),
                "Rep_Periods": 1,
                "Timesteps_per_Rep_Period": n_hours,
                "Sub_Weights": n_hours,
            })
        else:
            row.update({
                "Voll": "", "Demand_Segment": "",
                "Cost_of_Demand_Curtailment_per_MW": "",
                "Max_Demand_Curtailment": "",
                "$/MWh": "", "Rep_Periods": "",
                "Timesteps_per_Rep_Period": "", "Sub_Weights": "",
            })

        rows.append(row)

    meta_cols = ["Voll", "Demand_Segment", "Cost_of_Demand_Curtailment_per_MW",
                 "Max_Demand_Curtailment", "$/MWh", "Rep_Periods",
                 "Timesteps_per_Rep_Period", "Sub_Weights", "Time_Index"]
    load_cols = sorted(demand_cols.keys())
    col_order = meta_cols + load_cols

    df = pd.DataFrame(rows, columns=col_order)
    out_path = system_dir / "Demand_data.csv"
    df.to_csv(str(out_path), index=False)
    log.info(f"GenX demand file: {out_path}")


# ---------------------------------------------------------------------------
# OpenDSS loadshape generation — kvar from equipment-level power factors
# ---------------------------------------------------------------------------

# Reactive power is reconstructed per-hour from DECARB's ts.csv equipment
# breakdown (dem, hvac_HT, hvac_AC, pv) rather than from the original network's
# per-bus kvar/kw shapes.  Each component gets its own assumed power factor:
PF_COMPRESSOR = 0.85   # HVAC compressor (AC or heat-pump) — induction motor, lagging
PF_RESISTIVE = 1.0     # Electric resistance heat — purely resistive
PF_PV = 0.98           # Grid-tied inverter — near-unity
PF_BASELINE = 0.95     # Everything else (lighting, plug loads, fridge, water heat, EV)

HOURS_PER_YEAR = 8760
STEPS_PER_HOUR = {"hourly": 1, "15min": 4}


def study_timestep(study_dir):
    """Timestep a study was generated with, from the config snapshots saved in
    the study: config.yaml, or decarb.yaml for studies made before the
    timestep moved there. Falls back to the current config with a warning."""
    def _snapshot(name):
        path = Path(study_dir) / name
        if not path.exists():
            return None
        with open(path, encoding="utf-8-sig") as f:   # tolerate a BOM
            return (yaml.safe_load(f) or {}).get("timestep")

    for name in ("config.yaml", "decarb.yaml"):
        timestep = _snapshot(name)
        if timestep is not None:
            source = Path(study_dir) / name
            break
    else:
        log.warning(f"No timestep in the config snapshots of {study_dir}; "
                    f"using the current config")
        return resolve_timestep(load_config(), load_decarb_config())

    if timestep not in STEPS_PER_HOUR:
        raise ValueError(f"timestep in {source} must be one of "
                         f"{', '.join(STEPS_PER_HOUR)}, got {timestep!r}")
    return timestep


def select_period(values, step_indices, n_full_year, label):
    """Cut a DECARB output series to the configured date range.

    A full-year series is cut with step_indices; a series that already has
    exactly the date range's length (DECARB ran on a cut tm.csv) is used as is.
    Any other length is an error."""
    n_period = len(step_indices) if step_indices is not None else n_full_year
    if len(values) == n_full_year:
        return values[step_indices] if step_indices is not None else values
    if len(values) == n_period:
        return values
    expected = f"{n_full_year} (full year)"
    if step_indices is not None:
        expected += f" or {n_period} (date_range)"
    raise ValueError(f"{label}: {len(values)} rows, expected {expected} "
                     f"for this study's timestep")


def classify_heating_pf(decarb_case_dir, bldg_id):
    """Determine whether the building's electric heating is compressor-driven
    (heat pump → lagging PF) or resistive (baseboard/furnace → PF=1.0).

    Reads sp.csv to find the selected HVAC equipment, then checks hvac.csv
    to see if it has both heating and cooling capacity (heat pump) or
    heating-only with no compressor cooling (resistive).
    """
    sp_path = decarb_case_dir / str(bldg_id) / "in" / "sp.csv"
    hvac_path = decarb_case_dir / str(bldg_id) / "in" / "hvac.csv"

    try:
        sp_df = pd.read_csv(str(sp_path))
        hvac_df = pd.read_csv(str(hvac_path))
        selected = sp_df["pHVAC0"].iloc[0]
        row = hvac_df[hvac_df["ty"] == selected]
        if row.empty:
            return PF_COMPRESSOR
        pHVmx = float(row["pHVmx"].iloc[0])
        pACmx = float(row["pACmx"].iloc[0])
        # Resistive heat: has heating capacity but no compressor cooling
        return PF_RESISTIVE if (pHVmx > 0 and pACmx == 0) else PF_COMPRESSOR
    except (FileNotFoundError, KeyError, IndexError):
        return PF_COMPRESSOR


def compute_kvar_curve(ts_df, pf_ht):
    """Reconstruct reactive power from equipment-level power factors.

    kvar(t) = hvac_HT(t)*tan(acos(pf_ht))        — heating (resistive or compressor)
            + hvac_AC(t)*tan(acos(PF_COMPRESSOR))  — compressor cooling
            + baseline(t)*tan(acos(PF_BASELINE))   — everything else
            - pv(t)*tan(acos(PF_PV))               — near-unity inverter
    where baseline(t) = dem(t) - hvac_HT(t) - hvac_AC(t).
    """
    dem = ts_df["dem"].to_numpy(dtype=float)
    hvac_ht = ts_df["hvac_HT"].to_numpy(dtype=float)
    hvac_ac = ts_df["hvac_AC"].to_numpy(dtype=float)
    pv = ts_df["pv"].to_numpy(dtype=float)
    baseline = dem - hvac_ht - hvac_ac

    return (
        hvac_ht * np.tan(np.arccos(pf_ht))
        + hvac_ac * np.tan(np.arccos(PF_COMPRESSOR))
        + baseline * np.tan(np.arccos(PF_BASELINE))
        - pv * np.tan(np.arccos(PF_PV))
    )


# Non-residential (commercial) loads keep their SMART-DS profiles: per-unit,
# 15-min, one full year (35040 rows), standard time with no DST, at
# smartds.utc_offset_hours. They are shifted so row i covers the same UTC
# interval as the DECARB residential row i (ResStock: EST, UTC-5), resampled to
# the case timestep and cut to date_range.
RESSTOCK_UTC_OFFSET_HOURS = -5       # as create_decarb_inputs.RESSTOCK_UTC_OFFSET
SMARTDS_STEPS = 35040
FILE_REF = re.compile(r"file=([^\s\)]+)", re.IGNORECASE)


def smartds_shift_rows(cfg):
    """15-min rows to roll the SMART-DS profiles by to line them up with ResStock in UTC."""
    offset = (cfg.get("smartds") or {}).get("utc_offset_hours")
    if offset is None:
        raise KeyError("config.yaml is missing smartds.utc_offset_hours (clock of the SMART-DS "
                       "profiles, e.g. -6 for Full_Texas)")
    return int(round((RESSTOCK_UTC_OFFSET_HOURS - float(offset)) * 4))


def write_smartds_profile(src, dst, shift_rows, timestep, step_indices):
    """Shift, resample and cut one SMART-DS profile; write it to dst."""
    if not src.exists():
        raise FileNotFoundError(
            f"SMART-DS profile not found: {src}. Turn on data_acquisition.download_network_profiles "
            f"in config.yaml and run data acquisition.")
    values = np.loadtxt(src, delimiter=",", ndmin=1)
    if values.ndim > 1:
        values = values[:, -1]
    if len(values) != SMARTDS_STEPS:
        raise ValueError(f"{src}: {len(values)} rows, expected {SMARTDS_STEPS} (15-min, one year)")
    values = np.roll(values, shift_rows)
    rows_per_step = 4 // STEPS_PER_HOUR[timestep]
    values = values.reshape(-1, rows_per_step).mean(axis=1)
    if step_indices is not None:
        values = values[step_indices]
    np.savetxt(str(dst), values, fmt="%.6f")


def smartds_nonresidential(feeder_dir):
    """Non-residential Load lines of a SMART-DS feeder (kept as written) and the
    load shapes they use: {shape name: {"mult": file, "qmult": file or None}}."""
    from process_network_info import parse_dss_objects
    keep, used = [], set()
    for raw in (feeder_dir / "Loads.dss").read_text().splitlines():
        if not re.match(r"(?i)\s*new\s+load\.", raw):
            continue
        m = re.search(r"(?i)\byearly\s*=\s*(\S+)", raw)
        if m and m.group(1).lower().startswith("res"):
            continue
        keep.append(raw.strip())
        if m:
            used.add(m.group(1))
    shapes = {}
    if used:
        ls = parse_dss_objects(feeder_dir / "LoadShapes.dss", "Loadshape")
        for _, row in ls[ls["Name"].isin(used)].iterrows():
            files = {}
            for key in ("mult", "qmult"):
                ref = FILE_REF.search(str(row.get(key, "")))
                files[key] = Path(ref.group(1)).name if ref else None
            if files["mult"] is None:
                raise ValueError(f"{feeder_dir / 'LoadShapes.dss'}: no mult file for {row['Name']}")
            shapes[row["Name"]] = files
        missing = used - set(shapes)
        if missing:
            raise ValueError(f"{feeder_dir}: load shape(s) used in Loads.dss but not defined in "
                             f"LoadShapes.dss: {sorted(missing)[:5]}")
    return keep, shapes


def generate_opendss_loadshapes(decarb_case_dir, building_map_df,
                                network_dir, zone_name, output_dir, ncfg,
                                sub_map=None, timestep="hourly",
                                step_indices=None, smartds=None):
    """Build complete OpenDSS case directories from DECARB outputs.

    For each case:
      1. Write per-building kW/kvar profile CSVs into a profiles/ directory
      2. Copy template feeder folders (topology, lines, transformers)
      3. Rewrite LoadShapes.dss per feeder to define shapes pointing to our profiles
      4. Rewrite Loads.dss per feeder to reference our shape names
      5. Keep each feeder's non-residential loads, with their SMART-DS profiles
         shifted/resampled once per study into smartds["dst"] (see
         write_smartds_profile)

    The result is a self-contained case directory that run_opendss.py can
    compile and solve with no special logic.
    """
    # Feeder templates: the SMART-DS network downloaded into paths.network_dir
    zone_template = Path(network_dir) / zone_name / "scenarios" / "base_timeseries" / "opendss"
    if not zone_template.exists():
        raise FileNotFoundError(
            f"SMART-DS network for zone {zone_name} not found: {zone_template}. Turn on "
            f"data_acquisition.download_network in config.yaml and run data acquisition.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Time axis of the case: DECARB timestep, cut to date_range
    steps_per_hour = STEPS_PER_HOUR[timestep]
    n_full_year = HOURS_PER_YEAR * steps_per_hour
    n_steps = len(step_indices) if step_indices is not None else n_full_year
    interval_h = 1 / steps_per_hour
    stepsize = "1h" if steps_per_hour == 1 else f"{60 // steps_per_hour}m"

    # Use all buildings in the map (no zone filtering — matching already did that)
    feeder_df = building_map_df

    # ── Step 1: write per-building profile CSVs ──────────────────────────
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(exist_ok=True)

    # Raw matching file uses nrel_climate_match for building IDs
    bldg_col = "nrel_climate_match" if "nrel_climate_match" in feeder_df.columns else "bldg_id"
    unique_bldgs = feeder_df[bldg_col].dropna().astype(int).unique()
    pf_ht_cache = {}
    peak_kw = {}
    peak_kvar = {}
    processed = 0
    skipped = 0

    if sub_map is None:
        sub_map = {}
    substituted = 0

    for bldg_id in unique_bldgs:
        bldg_id = int(bldg_id)
        source_id = bldg_id  # which building's DECARB output to read
        ts_df = read_decarb_ts(decarb_case_dir, bldg_id)

        if ts_df is None or not {"buy", "sell", "dem", "hvac_HT", "hvac_AC", "pv"}.issubset(ts_df.columns):
            # Try substitution
            sub_id = sub_map.get(bldg_id)
            if sub_id is not None:
                ts_df = read_decarb_ts(decarb_case_dir, sub_id)
                if ts_df is not None and {"buy", "sell", "dem", "hvac_HT", "hvac_AC", "pv"}.issubset(ts_df.columns):
                    source_id = sub_id
                    substituted += 1
                else:
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue

        pf_ht = classify_heating_pf(decarb_case_dir, source_id)
        pf_ht_cache[bldg_id] = pf_ht

        net_kw = (ts_df["buy"].values - ts_df["sell"].values)
        net_kvar = compute_kvar_curve(ts_df, pf_ht)

        label = f"{decarb_case_dir.name}/{source_id} ts.csv"
        net_kw = select_period(net_kw, step_indices, n_full_year, label)
        net_kvar = select_period(np.asarray(net_kvar), step_indices, n_full_year, label)

        # Save under the original bldg_id so DSS file references resolve
        np.savetxt(str(profiles_dir / f"decarb_kw_{bldg_id}.csv"),
                   net_kw, fmt="%.6f")
        np.savetxt(str(profiles_dir / f"decarb_kvar_{bldg_id}.csv"),
                   net_kvar, fmt="%.6f")

        peak_kw[bldg_id] = float(np.max(np.abs(net_kw))) or 0.1
        peak_kvar[bldg_id] = float(np.max(np.abs(net_kvar))) or 0.01
        processed += 1

    log.info(f"Profiles: {processed} buildings written "
             f"({substituted} substituted), {skipped} skipped")

    # ── Step 2: copy template feeders and rewrite DSS files ──────────────
    def _clear_readonly(path):
        for root, dirs, files in os.walk(str(path)):
            for name in dirs + files:
                try:
                    os.chmod(os.path.join(root, name), stat.S_IWRITE)
                except OSError:
                    pass

    # Feeders in the template: all of the zone, or zone.substations / zone.feeders
    from process_network_info import list_feeders, select_feeders
    src_feeders = [fd for fd in select_feeders((smartds or {}).get("cfg", {}), list_feeders(zone_template))
                   if (fd / "Master.dss").exists()]

    log.info(f"Copying {len(src_feeders)} feeder templates into case directory")

    for src_feeder_dir in src_feeders:
        substation = src_feeder_dir.parent.name
        feeder = src_feeder_dir.name
        dst_feeder_dir = output_dir / substation / feeder

        # Copy template (topology files)
        if dst_feeder_dir.exists():
            _clear_readonly(dst_feeder_dir)
            shutil.rmtree(str(dst_feeder_dir))
        shutil.copytree(str(src_feeder_dir), str(dst_feeder_dir),
                        ignore=shutil.ignore_patterns("*.csv"))

        # Get loads for this feeder from the matching
        feeder_loads = feeder_df[
            (feeder_df["substation"] == substation) &
            (feeder_df["feeder"] == feeder)
        ] if "feeder" in feeder_df.columns else pd.DataFrame()

        # Collect which buildings are used by this feeder
        used_bldg_ids = set()
        load_lines = []
        shape_lines = []

        for _, row in feeder_loads.iterrows():
            bldg_id = int(row["nrel_climate_match"]) if "nrel_climate_match" in row.index else int(row["bldg_id"])
            load_name = row["Name"]

            if bldg_id not in peak_kw:
                # Building had no successful DECARB output — skip this load
                continue

            used_bldg_ids.add(bldg_id)
            shape_name = f"decarb_{bldg_id}"
            kw = peak_kw[bldg_id]
            kvar = peak_kvar[bldg_id]

            load_lines.append(
                f"New Load.{load_name} conn={row['conn']} bus1={row['bus1']} "
                f"kV={row['kv']} Vminpu={row['vminpu']} Vmaxpu={row['vmaxpu']} "
                f"model={row['model']} kW={kw} kvar={kvar} "
                f"Phases={row['phases']} yearly={shape_name}"
            )

        # Non-residential loads: SMART-DS lines as written, shapes on the
        # study's shifted/resampled profiles
        if smartds is not None:
            com_lines, com_shapes = smartds_nonresidential(src_feeder_dir)
            load_lines += com_lines
            for name, files in sorted(com_shapes.items()):
                for fn in files.values():
                    if fn and fn not in smartds["done"]:
                        write_smartds_profile(smartds["src"] / fn, smartds["dst"] / fn,
                                              smartds["shift"], timestep, step_indices)
                        smartds["done"].add(fn)
                line = (f"New Loadshape.{name} npts={n_steps} interval={interval_h:g} "
                        f"mult=(file={(smartds['dst'] / files['mult']).as_posix()})")
                if files["qmult"]:
                    line += f" qmult=(file={(smartds['dst'] / files['qmult']).as_posix()})"
                shape_lines.append(line)
            smartds["loads"] += len(com_lines)

        # Write LoadShapes.dss
        for bldg_id in sorted(used_bldg_ids):
            kw_file = (profiles_dir / f"decarb_kw_{bldg_id}.csv").as_posix()
            kvar_file = (profiles_dir / f"decarb_kvar_{bldg_id}.csv").as_posix()
            shape_lines.append(
                f"New Loadshape.decarb_{bldg_id} npts={n_steps} interval={interval_h:g} "
                f"useactual=true mult=(file={kw_file}) qmult=(file={kvar_file})"
            )

        (dst_feeder_dir / "LoadShapes.dss").write_text(
            "\n\n".join(shape_lines) + "\n")
        (dst_feeder_dir / "Loads.dss").write_text(
            "\n\n".join(load_lines) + "\n")

        # Rewrite Master.dss: solve at the case's timestep and length
        master_path = dst_feeder_dir / "Master.dss"
        master_text, n_replaced = re.subn(
            r"(?im)^\s*solve\s+mode\s*=\s*yearly\b.*$",
            f"Solve mode=yearly stepsize={stepsize} number={n_steps}",
            master_path.read_text())
        if n_replaced == 0:
            raise ValueError(f"No 'Solve mode=yearly ...' line to rewrite in {master_path}")
        master_path.write_text(master_text)

    log.info(f"OpenDSS case ready: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def find_study_dir(output_dir, study_num=None):
    """Find a study directory. If study_num is None, use the latest."""
    studies_root = output_dir / "studies"
    if not studies_root.exists():
        raise FileNotFoundError(f"No studies directory: {studies_root}")

    if study_num is not None:
        study_dir = studies_root / f"study_{study_num}"
        if not study_dir.exists():
            raise FileNotFoundError(f"Study not found: {study_dir}")
        return study_dir

    study_dirs = sorted(
        [d for d in studies_root.iterdir()
         if d.is_dir() and d.name.startswith("study_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    if not study_dirs:
        raise FileNotFoundError("No studies found")
    return study_dirs[-1]


def run_network_inputs(cfg, ncfg=None, study_num=None):
    if ncfg is None:
        ncfg = load_network_config()

    scope = cfg.get("scope", "statewide")
    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    network_dir = cfg["paths"]["network_dir"]

    study_dir = find_study_dir(output_dir, study_num)
    log.info(f"Using study: {study_dir.name}")

    # Save network config snapshot into study
    config_dir = Path(__file__).parents[1] / "config"
    src_ncfg = config_dir / "network.yaml"
    if src_ncfg.exists():
        shutil.copy2(str(src_ncfg), str(study_dir / "network.yaml"))

    # Load building map (study-level first, then global fallback)
    building_map_df = load_building_map(
        input_dir=input_dir, output_dir=output_dir, study_dir=study_dir)

    # Find all DECARB cases in the study
    decarb_root = study_dir / "decarb"
    if not decarb_root.exists():
        log.error(f"No decarb directory in study: {decarb_root}")
        return

    decarb_cases = [d.name for d in sorted(decarb_root.iterdir()) if d.is_dir()]
    if not decarb_cases:
        log.error("No DECARB cases found in study")
        return
    log.info(f"Found {len(decarb_cases)} DECARB cases: {decarb_cases}")

    # Studies made before the update_N level was removed keep each building's
    # files in {bldg_id}/update_0/; they are not supported.
    first_case = decarb_root / decarb_cases[0]
    if any((d / "update_0").is_dir() for d in first_case.iterdir() if d.is_dir()):
        raise ValueError(
            f"{study_dir.name} uses the old layout ({{bldg_id}}/update_0/in, out). "
            f"The pipeline now expects {{bldg_id}}/in and {{bldg_id}}/out; "
            f"regenerate the study with decarb-inputs.")

    if scope == "statewide":
        _run_genx(cfg, study_dir, decarb_root, decarb_cases,
                  building_map_df, ncfg)
    elif scope == "zone":
        zone_name = cfg["zone"]["name"]
        if not zone_name:
            log.error("scope=zone but no zone.name specified in config")
            return
        # OpenDSS needs the raw matching file (with Name, conn, bus1, etc.)
        # not the grouped building map
        feeder_parquet = study_dir / "feeder_summaries_matched.parquet"
        if not feeder_parquet.exists():
            feeder_parquet = input_dir / "feeder_summaries_matched.parquet"
        if not feeder_parquet.exists():
            log.error("No matching parquet found for OpenDSS case generation")
            return
        feeder_df = read_parquet(feeder_parquet)
        if "zone" in feeder_df.columns:
            file_zones = sorted(feeder_df["zone"].dropna().unique())
            if file_zones != [zone_name]:
                raise ValueError(
                    f"{feeder_parquet} is for zone(s) {', '.join(file_zones) or 'none'}, "
                    f"but zone.name in config.yaml is {zone_name!r}. Set zone.name to the "
                    f"study's zone, or use a study made for {zone_name}.")
        from process_network_info import check_matching_feeders
        check_matching_feeders(cfg, feeder_df, str(feeder_parquet))
        _run_opendss(cfg, study_dir, decarb_root, decarb_cases,
                     feeder_df, network_dir, zone_name, ncfg)
    else:
        log.error(f"Unknown scope: {scope}")


def _copy_base_case(base_case_dir, genx_case_dir):
    """Copy the pre-built GenX supply-side files into a study case directory."""
    if genx_case_dir.exists():
        shutil.rmtree(str(genx_case_dir))
    shutil.copytree(str(base_case_dir), str(genx_case_dir))
    log.info(f"Copied base case from {base_case_dir}")


def _run_genx(cfg, study_dir, decarb_root, decarb_cases, building_map_df, ncfg):
    genx_cfg = ncfg["genx"]
    network_mode = genx_cfg.get("network_mode", "1bus")
    hour_indices = resolve_hour_indices(cfg)
    n_hours = len(hour_indices) if hour_indices is not None else genx_cfg.get("n_hours", 8760)
    base_case_dir = genx_cfg.get("base_case")

    if base_case_dir and not Path(base_case_dir).exists():
        log.error(f"GenX base case directory not found: {base_case_dir}")
        return

    # Load bus mapping for multi-bus mode
    bus_mapping_df = None
    if network_mode != "1bus" and genx_cfg.get("bus_mapping"):
        bus_mapping_path = genx_cfg["bus_mapping"]
        if Path(bus_mapping_path).exists():
            bus_mapping_df = pd.read_csv(str(bus_mapping_path))

    substation_lookup = None
    if network_mode != "1bus":
        substation_lookup = build_substation_lookup(building_map_df)

    all_bldg_ids = building_map_df["bldg_id"].unique().tolist()
    genx_root = study_dir / "genx"

    for case_name in decarb_cases:
        decarb_case_dir = decarb_root / case_name
        genx_case_dir = genx_root / case_name

        log.info(f"GenX case: {case_name}")

        # Build substitution map for failed buildings
        sub_map = build_substitution_map(decarb_case_dir, all_bldg_ids, cfg)

        # Copy supply-side base case, then overwrite demand
        if base_case_dir:
            _copy_base_case(Path(base_case_dir), genx_case_dir)
        else:
            genx_case_dir.mkdir(parents=True, exist_ok=True)

        if network_mode == "1bus":
            demand = aggregate_demand_1bus(
                decarb_case_dir, building_map_df, n_hours,
                sub_map=sub_map, hour_indices=hour_indices)
        else:
            demand = aggregate_demand_multibus(
                decarb_case_dir, substation_lookup,
                bus_mapping_df, n_hours,
                sub_map=sub_map, hour_indices=hour_indices)

        write_genx_demand(demand, genx_case_dir, ncfg)


def _run_opendss(cfg, study_dir, decarb_root, decarb_cases,
                 feeder_df, network_dir, zone_name, ncfg):
    opendss_root = study_dir / "opendss"
    timestep = study_timestep(study_dir)
    step_indices = resolve_step_indices(cfg, STEPS_PER_HOUR[timestep])
    n_steps = len(step_indices) if step_indices is not None else HOURS_PER_YEAR * STEPS_PER_HOUR[timestep]
    log.info(f"OpenDSS cases at {timestep} resolution, {n_steps} steps"
             + (" (date_range)" if step_indices is not None else " (full year)"))

    # Get all building IDs from the matching file
    bldg_col = "nrel_climate_match" if "nrel_climate_match" in feeder_df.columns else "bldg_id"
    all_bldg_ids = feeder_df[bldg_col].dropna().astype(int).unique().tolist()

    from data_acquisition import smartds_profiles_dir
    smartds = {"cfg": cfg, "src": smartds_profiles_dir(cfg), "dst": opendss_root / "smartds_profiles",
               "shift": smartds_shift_rows(cfg), "done": set(), "loads": 0}
    smartds["dst"].mkdir(parents=True, exist_ok=True)
    log.info(f"Non-residential loads keep their SMART-DS profiles, shifted "
             f"{smartds['shift'] / 4:+g} h to the DECARB (ResStock) clock -> {smartds['dst']}")

    for case_name in decarb_cases:
        decarb_case_dir = decarb_root / case_name
        opendss_case_dir = opendss_root / case_name

        log.info(f"OpenDSS case: {case_name}")
        smartds["loads"] = 0

        # Build substitution map for failed buildings
        sub_map = build_substitution_map(decarb_case_dir, all_bldg_ids, cfg)

        generate_opendss_loadshapes(
            decarb_case_dir, feeder_df,
            network_dir, zone_name, opendss_case_dir, ncfg,
            sub_map=sub_map, timestep=timestep,
            step_indices=step_indices, smartds=smartds)
        log.info(f"  {smartds['loads']} non-residential loads kept "
                 f"({len(smartds['done'])} SMART-DS profiles prepared for this study)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate network inputs from DECARB results")
    args = ap.parse_args()

    cfg = load_config()
    ncfg = load_network_config()
    run_network_inputs(cfg, ncfg)
