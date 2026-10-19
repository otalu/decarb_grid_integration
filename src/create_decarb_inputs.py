import csv
import logging
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import requests

sys.path.insert(0, str(Path(__file__).parents[1]))
from internal_gains import equipment_kwh, lighting_kwh, occupant_count
from config.loader import (load_config, load_decarb_config, resolve_step_indices,
                           resolve_timestep)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FT2_TO_M2 = 0.092903
BTU_HR_TO_KW = 1 / 3412.14
OTHER_TEMPLATE_FILES = ["abp.csv", "pv.csv", "bess.csv", "wind.csv", "ev.csv"]

PV_TECH = "Tesla_Cash"
# Texas solar: ~1700-1800 kWh per kW installed per year
SOLAR_CAPACITY_FACTOR_ANNUAL = 1750.0  # kWh/kW/year


# ---------------------------------------------------------------------------
# Parquet reader
# ---------------------------------------------------------------------------

def read_parquet(path):
    try:
        return pd.read_parquet(str(path), engine="fastparquet")
    except Exception:
        return pd.read_parquet(str(path), engine="pyarrow")


# ---------------------------------------------------------------------------
# Elevation lookup
# ---------------------------------------------------------------------------

ELEVATION_MAX_RETRIES = 3
EPQS_NO_DATA = -1000000      # value EPQS returns for points with no elevation data


def fetch_elevation_usgs(lat, lon, max_retries=ELEVATION_MAX_RETRIES):
    """Return (elevation_m, None) on success or (None, error_message)."""
    url = f"https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&units=Meters&includeDate=false"
    error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            value = float(resp.json()["value"])
            if value <= EPQS_NO_DATA:
                return None, "USGS has no elevation data for this point"
            return value, None
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(2 * attempt)
    return None, error


def build_elevation_cache(unique_coords, max_workers=20):
    cache = {}
    failed = {}
    log.info(f"Fetching elevation for {len(unique_coords)} unique coordinates")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_elevation_usgs, lat, lon): (lat, lon)
            for lat, lon in unique_coords
        }
        for future in as_completed(futures):
            coord = futures[future]
            value, error = future.result()
            if error is None:
                cache[coord] = value
            else:
                failed[coord] = error

    if failed:
        examples = "\n".join(f"  lat={lat}, lon={lon}: {err}"
                             for (lat, lon), err in list(failed.items())[:10])
        raise RuntimeError(
            f"Elevation lookup failed for {len(failed)} of {len(unique_coords)} "
            f"coordinates (USGS EPQS, {ELEVATION_MAX_RETRIES} attempts each):\n{examples}\n"
            f"The service may be down or rate-limiting; try again later.")
    return cache


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

def load_templates(template_dir, outputs):
    templates = {}
    if outputs["bdg_i"]:
        templates["bdg_i"] = pd.read_csv(
            str(template_dir / "bdg_i.csv"), header=None, index_col=0, dtype=object)
    if outputs["cfg"]:
        templates["cfg"] = pd.read_csv(
            str(template_dir / "cfg.csv"), header=None, index_col=0, dtype=object)
    if outputs["other"]:
        for fname in OTHER_TEMPLATE_FILES:
            templates[fname] = pd.read_csv(
                str(template_dir / fname), header=None, index_col=0, dtype=object)
    if outputs["chp"] or outputs["sp"]:
        templates["chp"] = pd.read_csv(str(template_dir / "chp.csv"))
    if outputs["topo"]:
        templates["topo"] = pd.read_csv(str(template_dir / "topo.csv"))
    if outputs["hvac"] or outputs["sp"]:
        templates["hvac"] = pd.read_csv(str(template_dir / "hvac.csv"))
    if outputs["wh"] or outputs["sp"]:
        templates["wh"] = pd.read_csv(str(template_dir / "wh.csv"))
    if outputs["sp"]:
        templates["sp"] = pd.read_csv(str(template_dir / "sp.csv"))
    return templates


# ---------------------------------------------------------------------------
# Per-building data loaders
# ---------------------------------------------------------------------------

def load_consumption(bldg_id, input_dir, year, state="TX"):
    parquet = input_dir / "residential_profiles" / state / f"{bldg_id}-0.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Consumption file not found: {parquet}")
    df = read_parquet(parquet)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        if len(df) == 35040:
            df.index = pd.date_range(start=datetime(year, 1, 1, 0, 0), periods=35040, freq="15min")
        elif len(df) == 8760:
            df.index = pd.date_range(start=datetime(year, 1, 1, 0, 0), periods=8760, freq="h")
        else:
            raise ValueError(f"Unexpected consumption length for building {bldg_id}: {len(df)}")
    else:
        df.index = df.index - pd.Timedelta(minutes=15)
    return df


def load_setpoints(bldg_id, input_dir, year, state="TX"):
    parquet = input_dir / "setpoint_timeseries" / state / f"setpoint_{bldg_id}.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Setpoint file not found: {parquet}")
    df = read_parquet(parquet)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        if len(df) == 35040:
            df.index = pd.date_range(start=datetime(year, 1, 1, 0, 0), periods=35040, freq="15min")
        elif len(df) == 8760:
            df.index = pd.date_range(start=datetime(year, 1, 1, 0, 0), periods=8760, freq="h")
        else:
            raise ValueError(f"Unexpected setpoint length for building {bldg_id}: {len(df)}")
    else:
        df.index = df.index - pd.Timedelta(minutes=15)
    return df


def load_schedule(bldg_id, input_dir, state="TX"):
    parquet = input_dir / "schedules" / state / f"schedule_{bldg_id}.parquet"
    if not parquet.exists():
        return None
    df = read_parquet(parquet)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    return df


def get_building_row(bldg_id, metadata_df):
    if bldg_id in metadata_df.index:
        return metadata_df.loc[[bldg_id]]
    if "bldg_id" in metadata_df.columns:
        row = metadata_df[metadata_df["bldg_id"] == bldg_id]
    elif "building_id" in metadata_df.columns:
        row = metadata_df[metadata_df["building_id"] == bldg_id]
    else:
        row = metadata_df[metadata_df.index.astype(str) == str(bldg_id)]
    if len(row) == 0:
        raise ValueError(f"Building {bldg_id} not found in metadata")
    return row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resample_if_needed(series, timestep, method="sum"):
    if timestep == "hourly" and len(series) == 35040:
        if method == "sum":
            resampled = series.resample("h").sum()
        elif method == "mean":
            resampled = series.resample("h").mean()
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        if len(resampled) > 8760:
            resampled = resampled.iloc[:8760]
        return resampled
    return series


def is_heat_pump_building(building_row):
    heating_eff = str(building_row["in.hvac_heating_efficiency"].iloc[0])
    return any(kw in heating_eff for kw in ("ASHP", "MSHP", "Heat Pump"))


def get_resstock_capacities(building_row):
    heating_btu = float(building_row["out.capacity.heating..btu_per_hr"].iloc[0])
    cooling_btu = float(building_row["out.capacity.cooling..btu_per_hr"].iloc[0])
    return heating_btu * BTU_HR_TO_KW, cooling_btu * BTU_HR_TO_KW


def compute_pv_panel_count(consumption_df, panel_rating_kw):
    """Size PV to offset ~100% of annual electricity consumption.

    Parameters
    ----------
    consumption_df : DataFrame
        Building consumption timeseries (hourly or 15-min).
    panel_rating_kw : float
        Per-panel capacity from pv.csv template (e.g. 0.34 kW for Tesla_Cash).

    Returns
    -------
    int
        Number of panels (minimum 1).
    """
    elec_cols = [c for c in consumption_df.columns
                 if "electricity" in c.lower() and "total" in c.lower()
                 and "energy_consumption" in c.lower()]
    if elec_cols:
        annual_kwh = float(consumption_df[elec_cols[0]].sum())
    else:
        # Fallback: sum all electricity end-use columns
        elec_cols = [c for c in consumption_df.columns
                     if c.startswith("out.electricity.") and "kwh" in c.lower()]
        if elec_cols:
            annual_kwh = float(consumption_df[elec_cols].sum().sum())
        else:
            annual_kwh = 10000.0  # conservative fallback
    system_kw = annual_kwh / SOLAR_CAPACITY_FACTOR_ANNUAL
    return max(1, int(np.ceil(system_kw / panel_rating_kw)))


def compute_required_pv_area(panel_count, panel_area_m2):
    """Roof area for pBpv in bdg_i.csv.  DECARB constrains ar*zPV <= pBpv
    (pv_panels.jl), so set enough headroom for the panels we install."""
    return (panel_count + 1) * panel_area_m2


def compute_optimal_orientation(lat, lon, alt, timestep, tm_df):
    """Optimal fixed-panel tilt/azimuth from the building's own solar resource.

    Ported from create_pv_sp_case.py — uses pvlib SPA solar position weighted
    by direct normal irradiance so the orientation favours hours with strong sun.
    Solar position is taken at the start of each period (tm.csv pP are UTC
    period ends).
    """
    times = (pd.to_datetime(tm_df["pP"]) - TIMESTEP_LENGTH[timestep]).dt.tz_localize("UTC")
    sdni = tm_df["pSdni"].to_numpy()

    solpos = pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex(times), latitude=lat, longitude=lon, altitude=alt
    )
    elevation = np.maximum(solpos["apparent_elevation"].to_numpy(), 0.0)
    azimuth = solpos["azimuth"].to_numpy()

    weights = np.maximum(sdni, 0.0)
    if weights.sum() <= 0:
        log.warning(f"No direct normal irradiance (pSdni) at lat={lat}, lon={lon}; "
                    f"using fallback PV orientation tilt=35, azimuth=180")
        return 35.0, 180.0  # fallback

    mean_elevation = float(np.sum(weights * elevation) / weights.sum())
    azimuth_rad = np.radians(azimuth)
    sin_mean = np.sum(weights * np.sin(azimuth_rad)) / weights.sum()
    cos_mean = np.sum(weights * np.cos(azimuth_rad)) / weights.sum()
    mean_azimuth = float(np.degrees(np.arctan2(sin_mean, cos_mean)) % 360)
    tilt = float(np.clip(90.0 - mean_elevation, 0.0, 90.0))
    return tilt, mean_azimuth


# ---------------------------------------------------------------------------
# File generators
# ---------------------------------------------------------------------------

TIMESTEP_LENGTH = {"15min": pd.Timedelta(minutes=15), "hourly": pd.Timedelta(hours=1)}

# ResStock timestamps are period ends in Eastern Standard Time (UTC-5) for
# every state, no daylight saving. DECARB reads tm.csv timestamps as UTC
# period ends (DECARB docs/inputs_outputs.md).
RESSTOCK_UTC_OFFSET = pd.Timedelta(hours=5)


def tm_utc_period_ends(year, timestep):
    """UTC period-end timestamps for a full year of ResStock data."""
    step = TIMESTEP_LENGTH[timestep]
    n_steps = 8760 if timestep == "hourly" else 35040
    return pd.date_range(start=pd.Timestamp(year, 1, 1) + step + RESSTOCK_UTC_OFFSET,
                         periods=n_steps, freq=step)


def local_period_starts(utc_ends, timestep, tz):
    """Local clock time (with daylight saving) at the start of each period."""
    starts = utc_ends - TIMESTEP_LENGTH[timestep]
    return starts.tz_localize("UTC").tz_convert(tz).tz_localize(None)


def generate_tm_data(consumption_df, setpoint_df, building_row, schedule_df,
                     tariff, dcfg):
    timestep = dcfg["timestep"]
    year = _worker_state["year"]

    hvac_data = {
        col: building_row[col].iloc[0] if col in building_row.columns else None
        for col in ("in.heating_unavailable_period", "in.cooling_unavailable_period",
                    "in.hvac_cooling_type", "in.hvac_heating_type")
    }

    timestamps = tm_utc_period_ends(year, timestep)
    # Clock-time rules (TOU peak hours, unavailable periods) use local time
    local = local_period_starts(timestamps, timestep, _worker_state["cfg"]["timezone"])

    df = pd.DataFrame(index=timestamps)
    df.index.name = "pP"

    # Tariff
    if tariff["type"] == "flat":
        df["pQcostBuy"] = tariff["rate"]
    elif tariff["type"] == "tou":
        off_peak_price = tariff["rate"]
        peak_price = tariff["peak_price"]
        ts_times = local.time
        # Support multiple peak windows (list of {start, end} dicts)
        peak_windows = tariff.get("peak_windows", [])
        if not peak_windows and "peak_start" in tariff:
            # Backward compat: single peak_start/peak_end
            peak_windows = [{"start": tariff["peak_start"], "end": tariff["peak_end"]}]
        is_peak = pd.Series(False, index=df.index)
        for win in peak_windows:
            ws = datetime.strptime(win["start"], "%H:%M").time()
            we = datetime.strptime(win["end"], "%H:%M").time()
            if ws < we:
                is_peak |= (ts_times >= ws) & (ts_times < we)
            else:
                is_peak |= (ts_times >= ws) | (ts_times < we)
        df["pQcostBuy"] = off_peak_price
        df.loc[is_peak, "pQcostBuy"] = peak_price

    df["pQcostSell"] = tariff.get("sell_rate", 0.18)

    # Capacity charge — applies to ALL hours when enabled (pQmx=1 everywhere)
    cap_cfg = tariff.get("capacity_charge", dcfg.get("capacity_charge", {"enabled": False}))
    if cap_cfg.get("enabled", False):
        df["pQmx"] = 1
        df["pQmxCost"] = cap_cfg["charge_value"]
    else:
        df["pQmx"] = 0
        df["pQmxCost"] = 0.0

    fuel = dcfg["fuel_costs"]
    df["pQco2"] = fuel["co2"]
    df["pGcost"] = fuel["gas"]
    df["pLcost"] = fuel["liquid"]

    # Lighting and equipment are electricity only: DECARB bills pQlight and
    # pQequip as electricity demand. Other gas/propane/oil end uses (range,
    # dryer, fireplace, gas lighting) are not modeled. Same definitions as the
    # thermal model's internal gains (internal_gains.py).
    lighting_15min = lighting_kwh(consumption_df)
    if timestep == "15min":
        df["pQlight"] = lighting_15min.values * 4
    else:
        df["pQlight"] = resample_if_needed(lighting_15min, timestep, "sum").values

    equip_kwh = resample_if_needed(equipment_kwh(consumption_df), timestep, "sum").values
    df["pQequip"] = equip_kwh if timestep == "hourly" else equip_kwh * 4

    # Outdoor temperature
    temp_col = "out.outdoor_air_drybulb_temp..c"
    if temp_col not in consumption_df.columns:
        raise ValueError(f"Outdoor temperature column not found: {temp_col}")
    df["pTout"] = resample_if_needed(consumption_df[temp_col], timestep, "mean").values

    # Comfort bounds from setpoints
    if setpoint_df is not None and "max_comfort_temp" in setpoint_df.columns:
        df["pTmx"] = resample_if_needed(setpoint_df["max_comfort_temp"], timestep, "mean").values
        df["pTmn"] = resample_if_needed(setpoint_df["min_comfort_temp"], timestep, "mean").values
    else:
        df["pTmx"] = 26.0
        df["pTmn"] = 18.0

    # Mode from schedule
    if schedule_df is not None and "schedule" in schedule_df.columns:
        mode_series = schedule_df["schedule"]
        if timestep == "hourly" and len(mode_series) == 35040:
            mode_series = resample_if_needed(mode_series, timestep, "mean").round()
    else:
        mode_series = pd.Series(0, index=df.index)

    df["pTon"] = 1

    def apply_unavail(unavail_str, mode_vals, mode_match):
        if not unavail_str or str(unavail_str).lower() in ("never", "nan", "none"):
            return
        if str(unavail_str).lower() in ("year round", "year-round"):
            df.loc[(mode_vals == mode_match) | (mode_vals == 0), "pTon"] = 0
            return
        try:
            parts = str(unavail_str).split("-")
            if len(parts) == 2:
                s = datetime.strptime(f"{parts[0].strip()} {year}", "%b %d %Y")
                e = datetime.strptime(f"{parts[1].strip()} {year}", "%b %d %Y")
                if e < s:
                    period_mask = (
                        (local.month > s.month) |
                        ((local.month == s.month) & (local.day >= s.day)) |
                        (local.month < e.month) |
                        ((local.month == e.month) & (local.day <= e.day))
                    )
                else:
                    period_mask = (
                        ((local.month > s.month) |
                         ((local.month == s.month) & (local.day >= s.day))) &
                        ((local.month < e.month) |
                         ((local.month == e.month) & (local.day <= e.day)))
                    )
                season_mask = (mode_vals == mode_match) | (mode_vals == 0)
                df.loc[period_mask & season_mask, "pTon"] = 0
        except Exception:
            pass

    apply_unavail(hvac_data.get("in.heating_unavailable_period"), mode_series.values, 1)
    apply_unavail(hvac_data.get("in.cooling_unavailable_period"), mode_series.values, -1)

    # If building has no cooling equipment, mark cooling as unavailable year-round
    cooling_type = str(hvac_data.get("in.hvac_cooling_type", "") or "").strip().lower()
    if cooling_type in ("none", ""):
        cooling_mode_mask = (mode_series.values == -1) | (mode_series.values == 0)
        df.loc[cooling_mode_mask, "pTon"] = 0

    # If building has no heating equipment, mark heating as unavailable year-round
    heating_type = str(hvac_data.get("in.hvac_heating_type", "") or "").strip().lower()
    if heating_type in ("none", ""):
        heating_mode_mask = (mode_series.values == 1) | (mode_series.values == 0)
        df.loc[heating_mode_mask, "pTon"] = 0

    # Hot water demand
    hw_load_cols = [
        "out.load.hot_water.energy_delivered..kbtu",
        "out.load.hot_water_solar_thermal..kbtu",
        "out.load.hot_water_tank_losses..kbtu",
    ]
    avail = [c for c in hw_load_cols if c in consumption_df.columns]
    if avail:
        hw_kbtu = consumption_df[avail].sum(axis=1) * 0.293071
        hw_dem = resample_if_needed(hw_kbtu, timestep, "sum").values
        # Tank losses go negative when the tank's surroundings are warmer than
        # the water (e.g. TX garages/attics); DECARB requires demand >= 0.
        neg = hw_dem < 0
        if neg.any():
            bid = building_row.index[0] if building_row.index.name in ("bldg_id", "building_id") else "?"
            log.info(f"Building {bid}: hot water demand negative in {int(neg.sum())} steps "
                     f"(min {hw_dem.min():.4f} kWh), clipped to 0")
        df["pHWdem"] = np.clip(hw_dem, 0, None)
    else:
        df["pHWdem"] = 0.0

    # Solar radiation
    for out_col, src_col, scale in [
        ("pSdni", "out.weather.direct_normal_solar_radiation..watt_per_m2", 1000.0),
        ("pSdhi", "out.weather.diffuse_solar_radiation..watt_per_m2", 1000.0),
    ]:
        if src_col in consumption_df.columns:
            df[out_col] = resample_if_needed(
                consumption_df[src_col], timestep, "mean").values / scale
        else:
            df[out_col] = 0.0

    # Wind
    wind_col = "out.weather.wind_speed..meter_per_second"
    df["pWms"] = (
        resample_if_needed(consumption_df[wind_col], timestep, "mean").values
        if wind_col in consumption_df.columns else 0.0
    )

    # EV placeholders
    df["pEVdem"] = 0.0
    df["pEVtm"] = 0.0
    df["pEVtype"] = 0

    # Occupancy
    occ_sched_col = "out.schedules.occupants"
    occ_meta_col = "in.occupants"
    if occ_sched_col in consumption_df.columns:
        occ_sched = resample_if_needed(consumption_df[occ_sched_col], timestep, "mean")
        if occ_meta_col in building_row.columns:
            total = occupant_count(building_row[occ_meta_col].iloc[0])
            df["pBppl"] = (occ_sched.fillna(0).values * total).round().astype(int)
        else:
            df["pBppl"] = occ_sched.fillna(0).round().astype(int)
    else:
        df["pBppl"] = 0

    steps_per_hour = 1 if timestep == "hourly" else 4
    step_indices = resolve_step_indices(_worker_state["cfg"], steps_per_hour)
    if step_indices is not None:
        df = df.iloc[step_indices]

    return df


def create_tm_file(tm_df, output_dir):
    out = tm_df.copy()
    out.index = out.index.strftime("%Y-%m-%dT%H:%M")   # UTC period ends
    out.index.name = "pP"
    out.to_csv(str(output_dir / "tm.csv"), sep=",")


def create_bdg_i_file(output_dir, template, building_row,
                      elevation_cache, pBhvac=None,
                      pv_roof_area=None, pv_tilt=None, pv_azimuth=None):
    df = template.copy()

    def set_val(key, value):
        if key in df.index:
            df.loc[key, df.columns[0]] = value

    # Time zone of DECARB's output dates (tm.csv itself is UTC)
    set_val("pBtz", _worker_state["cfg"]["timezone"])

    lat = float(building_row["in.weather_file_latitude"].iloc[0])
    lon = float(building_row["in.weather_file_longitude"].iloc[0])
    set_val("pBalt", elevation_cache[(lat, lon)])
    set_val("pBlat", lat)
    set_val("pBlon", lon)

    sqft = float(building_row["in.sqft..ft2"].iloc[0])
    set_val("pBfoot", sqft * FT2_TO_M2)

    wall_ft2 = float(building_row["out.params.wall_area_above_grade_exterior..ft2"].iloc[0])
    wall_per_side = (wall_ft2 * FT2_TO_M2) / 4.0
    for col in ("pBwest", "pBeast", "pBnorth", "pBsouth"):
        set_val(col, wall_per_side)

    if pBhvac is not None:
        set_val("pBhvac", pBhvac)

    # PV roof area, tilt, azimuth — set when solar is enabled
    if pv_roof_area is not None:
        set_val("pBpv", pv_roof_area)
    if pv_tilt is not None:
        set_val("pBtilt", pv_tilt)
    if pv_azimuth is not None:
        set_val("pBazi", pv_azimuth)

    df.to_csv(str(output_dir / "bdg_i.csv"), header=False)


TIMESTEP_FIT_STEPS = {"15min": 1, "hourly": 4}   # 15-min ResStock rows per DECARB step

THERMAL_FIT_HOURS = 0.25   # the thermal model is fitted on 15-min steps


def create_bdg_ii_file(output_dir, bldg_id, regression_coeffs):
    """Coefficients as fitted; DECARB converts them to its step length using
    pBtime (thermal_response_coefficients in thermal_load.jl)."""
    key = str(bldg_id)
    if key not in regression_coeffs:
        raise KeyError(f"Building {bldg_id} not found in the thermal model results")
    coeffs = regression_coeffs[key]
    pd.DataFrame([{
        "pBk1": coeffs["k1"],
        "pBk2": coeffs["k2"] * 1000,
        "pBk3": coeffs["k3"] * 1000,
        "pBtime": THERMAL_FIT_HOURS,
    }]).to_csv(str(output_dir / "bdg_ii.csv"), index=False)


def create_cfg_file(output_dir, template, consumption_df, tm_df, timestep):
    """cfg.csv: pP0 is the start of the first tm.csv period (tm.csv timestamps
    are period ends), pTin0 the ResStock indoor temperature at that moment."""
    df = template.copy()
    col = df.columns[0]
    if "pP0" not in df.index:
        raise KeyError("cfg.csv template has no pP0 row")
    df.loc["pP0", col] = (tm_df.index[0] - TIMESTEP_LENGTH[timestep]).strftime("%Y-%m-%dT%H:%M")

    temp_col = "out.indoor_temperature.conditioned_space..c"
    if temp_col in consumption_df.columns and "pTin0" in df.index:
        # Last 15-min ResStock row before the first period; the first row
        # when the period starts on Jan 1 (no earlier row).
        steps_per_hour = 1 if timestep == "hourly" else 4
        step_indices = resolve_step_indices(_worker_state["cfg"], steps_per_hour)
        first_step = 0 if step_indices is None else int(step_indices[0])
        row = max(first_step * TIMESTEP_FIT_STEPS[timestep] - 1, 0)
        df.loc["pTin0", col] = consumption_df[temp_col].iloc[row]
    df.to_csv(str(output_dir / "cfg.csv"), header=False)


def create_other_files(output_dir, templates, has_solar=False):
    for fname in OTHER_TEMPLATE_FILES:
        if fname == "pv.csv" and not has_solar:
            df = templates[fname].copy()
            # Zero out mx (col 2) for data rows, skip header row at index "ty"
            data_mask = df.index != "ty"
            df.loc[data_mask, 2] = 0
            df.to_csv(str(output_dir / fname), header=False)
        else:
            templates[fname].to_csv(str(output_dir / fname), header=False)


def _build_chp_df(template, bldg_id, hp_summary_df, building_row=None):
    df = template.copy()
    if bldg_id not in hp_summary_df.index:
        raise ValueError(f"Building {bldg_id} not found in heat pump summary")
    row = hp_summary_df.loc[[bldg_id]]
    hp_heating_kw = float(row["heating_capacity_kw"].iloc[0])

    heating_kw = hp_heating_kw

    zeros_mask = df["ty"].astype(str).isin(["0", "0.0"])
    zeros_row = df[zeros_mask].copy()
    df = df[~zeros_mask].copy()
    df["mx"] = heating_kw
    return pd.concat([df, zeros_row], ignore_index=True)


def comfort_reference_temps(setpoint_df):
    """Average comfort-band midpoint over the building's heating-mode and
    cooling-mode periods (setpoint 'mode' 1 / -1). Falls back to the
    all-year average for a mode the building never enters."""
    mid = (setpoint_df["min_comfort_temp"] + setpoint_df["max_comfort_temp"]) / 2
    refs = []
    for mode_value in (1, -1):
        sel = mid[setpoint_df["mode"] == mode_value]
        refs.append(float(sel.mean()) if len(sel) else float(mid.mean()))
    return refs[0], refs[1]


def capacity_for_indoor_reference(capacity_kw, loss, t_design, t_indoor, mode):
    """Re-express a capacity curve fitted against the design temperature
    (heating: C*(1 - k*(Td - Tout)); cooling: C*(1 - k*(Tout - Td))) in
    DECARB's form, which derates against the indoor temperature
    (A*(1 - k'*(Tin - Tout)) / A*(1 - k'*(Tout - Tin))). Exact when the
    indoor temperature equals t_indoor. Returns (A, k')."""
    if mode == "heating":
        factor = 1 + loss * (t_indoor - t_design)
    else:
        factor = 1 - loss * (t_indoor - t_design)
    if factor <= 0:
        raise ValueError(f"{mode} capacity conversion factor {factor:.3f} <= 0 "
                         f"(loss={loss}, design={t_design}, indoor={t_indoor})")
    return capacity_kw * factor, loss / factor


def _build_hvac_df(template, bldg_id, hp_summary_df, building_row=None,
                   setpoint_df=None):
    df = template.copy()
    if bldg_id not in hp_summary_df.index:
        raise ValueError(f"Building {bldg_id} not found in heat pump summary")
    row = hp_summary_df.loc[[bldg_id]]
    count = float(row["count"].iloc[0])
    # Use rated capacity at design temp — DECARB applies its own temperature
    # derating via pHVmx_ and the indoor/outdoor temperature difference.
    # Previously we wrote the pre-derated capacity (at the sizing outdoor temp),
    # causing DECARB to double-derate and severely undersize the HP.
    if "rated_heating_capacity_kw" in row.columns:
        hp_heating_kw = float(row["rated_heating_capacity_kw"].iloc[0]) * count
        hp_cooling_kw = float(row["rated_cooling_capacity_kw"].iloc[0]) * count
    else:
        # Fallback for older hp_summary files without rated columns
        hp_heating_kw = float(row["heating_capacity_kw"].iloc[0]) * count
        hp_cooling_kw = float(row["cooling_capacity_kw"].iloc[0]) * count

    heating_kw = hp_heating_kw
    cooling_kw = hp_cooling_kw

    zeros_mask = df["ty"].astype(str).isin(["0", "0.0"])
    zeros_row = df[zeros_mask].copy()
    df = df[~zeros_mask].copy()
    df.loc[df["pHVmx"] != 0, "pHVmx"] = heating_kw
    df.loc[df["pACmx"] != 0, "pACmx"] = cooling_kw

    # The NEEP capacity fit derates from the design temperature; DECARB
    # derates capacity from the indoor temperature. Re-express the fitted
    # curve for each building's average heating/cooling comfort midpoint.
    if setpoint_df is None:
        raise ValueError(f"Building {bldg_id}: setpoint file needed to convert "
                         f"heat pump capacity derating for DECARB")
    t_design = float(row["design_temperature_c"].iloc[0])
    t_heat, t_cool = comfort_reference_temps(setpoint_df)
    hp_heating_kw, heating_loss = capacity_for_indoor_reference(
        hp_heating_kw, float(row["heating_capacity_loss"].iloc[0]), t_design, t_heat, "heating")
    hp_cooling_kw, cooling_loss = capacity_for_indoor_reference(
        hp_cooling_kw, float(row["cooling_capacity_loss"].iloc[0]), t_design, t_cool, "cooling")

    hp_index = str(row["hp_index"].iloc[0])
    mitsubishi_row = {
        "ty": f"Mitsubishi ASHP {hp_index}",
        "pHVmx": hp_heating_kw,
        "pACmx": hp_cooling_kw,
        "pHVeff": float(row["cop_heating"].iloc[0]),
        "pACeff": float(row["cop_cooling"].iloc[0]),
        "temp": t_design,
        "pHVmx_": heating_loss,
        "pACmx_": cooling_loss,
        "pHVeff_": float(row["cop_loss_heating"].iloc[0]),
        "pACeff_": float(row["cop_loss_cooling"].iloc[0]),
        "inv": float(row["capital_cost"].iloc[0]),
        "fom": 0,
        "vom": 0,
        "life": 15,
    }
    return pd.concat([df, pd.DataFrame([mitsubishi_row]), zeros_row], ignore_index=True)


def create_hvac_file(output_dir, template, bldg_id, hp_summary_df,
                     building_row=None, setpoint_df=None):
    df = _build_hvac_df(template, bldg_id, hp_summary_df, building_row, setpoint_df)
    df.to_csv(str(output_dir / "hvac.csv"), index=False)
    return df


def create_chp_file(output_dir, template, bldg_id, hp_summary_df,
                    building_row=None):
    df = _build_chp_df(template, bldg_id, hp_summary_df, building_row)
    df.to_csv(str(output_dir / "chp.csv"), index=False)
    return df


def _resolve_identifier(original, candidates, rng, bldg_id, role,
                        fuel_scoped=False):
    import difflib
    if original in candidates:
        return original
    close = difflib.get_close_matches(original, candidates, n=1, cutoff=0.6)
    if close:
        return close[0]
    pool = candidates
    if fuel_scoped:
        fuel = original.split()[0] if original.split() else ""
        fuel_pool = [c for c in candidates if str(c).startswith(fuel)]
        if fuel_pool:
            pool = fuel_pool
    return rng.choice(pool)


def create_sp_file(output_dir, template, building_row,
                   hvac_df, chp_df, wh_df,
                   elec_level=0, has_solar=False,
                   hp_name=None, panel_count=0):
    """Generate sp.csv — starting-point equipment for DECARB.

    Parameters
    ----------
    elec_level : int
        Electrification level (0 or 100). When > 0 and building has fossil
        heating, pHVAC0 is forced to the Mitsubishi ASHP heat pump.
    has_solar : bool
        If True, pPVz0 is set to panel_count and pPVyn to "NO" (fixed).
    hp_name : str or None
        Mitsubishi ASHP name from hp_summary (e.g. "Mitsubishi ASHP 3").
        Used when elec_level > 0.
    panel_count : int
        Number of PV panels to install (used when has_solar=True).
    """
    import random
    rng = random.Random(42)

    bldg_id = (building_row.index[0]
               if building_row.index.name in ("bldg_id", "building_id")
               else building_row.iloc[0].get("bldg_id",
                    building_row.iloc[0].get("building_id", "?")))
    df = template.copy()
    zeros_mask = df["pCHP0"].astype(str).isin(["0", "0.0"])
    zeros_row = df[zeros_mask].copy()
    df = df[~zeros_mask].copy()

    heating_fuel = str(building_row["in.heating_fuel"].iloc[0])
    heating_eff = str(building_row["in.hvac_heating_efficiency"].iloc[0])
    heating_id = f"{heating_fuel} {heating_eff}".replace(",", "")
    cooling_id = str(building_row["in.hvac_cooling_efficiency"].iloc[0]).replace(",", "")
    wh_id = str(building_row["in.water_heater_efficiency"].iloc[0])

    HEATING_REMAPS = {
        "Electricity MSHP SEER 14.5 8.2 HSPF": "Electricity ASHP SEER 15 8.5 HSPF",
        "Electricity MSHP SEER 29.3 14 HSPF": "Electricity ASHP SEER 24 12 HSPF",
    }
    if heating_id in HEATING_REMAPS:
        heating_id = HEATING_REMAPS[heating_id]

    if cooling_id == "Non-Ducted Heat Pump":
        cooling_id = "Ducted Heat Pump"

    if "Heat Pump" in wh_id:
        wh_id = "Electric Heat Pump 80 gal"

    heating_none = heating_id.strip().lower().endswith("none")
    if "Shared Heating" in heating_id:
        fuel = heating_id.split()[0] if heating_id.split() else ""
        hvac_heat_only = hvac_df[
            (hvac_df["ty"].notna()) &
            (hvac_df["ty"].astype(str).str.lower() != "none") &
            (hvac_df["pHVmx"] > 0) &
            (hvac_df["pACmx"] == 0) &
            (hvac_df["ty"].astype(str).str.startswith(fuel))
        ]["ty"].tolist()
        chp_heat_only = chp_df[
            (chp_df["ty"].notna()) &
            (chp_df["ty"].astype(str).str.lower() != "none") &
            (chp_df["ty"].astype(str).str.startswith(fuel))
        ]["ty"].tolist()
        pool = hvac_heat_only + chp_heat_only
        if not pool:
            pool = hvac_df[
                (hvac_df["ty"].notna()) &
                (hvac_df["pHVmx"] > 0) & (hvac_df["pACmx"] == 0)
            ]["ty"].tolist() + chp_df[
                (chp_df["ty"].notna()) &
                (chp_df["ty"].astype(str).str.lower() != "none")
            ]["ty"].tolist()
        if pool:
            heating_id = rng.choice(pool)

    cooling_none = cooling_id.strip().lower() == "none"
    cooling_ducted_hp = cooling_id == "Ducted Heat Pump"
    cooling_shared = "Shared Cooling" in cooling_id

    if cooling_shared:
        pool = hvac_df[
            (hvac_df["ty"].notna()) &
            (hvac_df["ty"].astype(str).str.lower() != "none") &
            (hvac_df["pACmx"] > 0) &
            (hvac_df["pHVmx"] == 0)
        ]["ty"].tolist()
        if pool:
            cooling_id = rng.choice(pool)

    all_hvac_chp = [t for t in hvac_df["ty"].tolist() + chp_df["ty"].tolist()
                    if t and str(t).lower() != "none"]
    valid_hvac = [t for t in hvac_df["ty"].values
                  if t and str(t).lower() != "none" and "Ducted Heat Pump" not in str(t)]
    valid_wh = [t for t in wh_df["ty"].values if t and str(t).lower() != "none"]

    if not heating_none:
        heating_in_hvac = heating_id in hvac_df["ty"].values
        heating_in_chp = heating_id in chp_df["ty"].values
        if not heating_in_hvac and not heating_in_chp:
            heating_id = _resolve_identifier(
                heating_id, all_hvac_chp, rng, bldg_id, "heating", fuel_scoped=True)

    if not cooling_none and not cooling_ducted_hp:
        if cooling_id not in hvac_df["ty"].values:
            cooling_id = _resolve_identifier(cooling_id, valid_hvac, rng, bldg_id, "cooling")

    if wh_id not in wh_df["ty"].values:
        wh_id = _resolve_identifier(wh_id, valid_wh, rng, bldg_id, "wh")

    df["pCHPz0"] = 0
    df["pCHPyn"] = "NO"
    df["pHVACz0"] = 0
    df["pHVACyn"] = "NO"

    # --- Electrification override ---
    # When elec_level > 0 and building has fossil heating, force the
    # Mitsubishi ASHP as the starting HVAC (both heating + cooling).
    is_fossil = not is_heat_pump_building(building_row)
    if elec_level > 0 and is_fossil and hp_name:
        # The ASHP handles both heating and cooling — single unit in sp.csv
        if hp_name in df["pHVAC0"].values:
            df.loc[df["pHVAC0"] == hp_name, "pHVACz0"] = 1
            df.loc[df["pHVAC0"] == hp_name, "pHVACyn"] = "NO"
        else:
            df.iloc[0, df.columns.get_loc("pHVAC0")] = hp_name
            df.iloc[0, df.columns.get_loc("pHVACz0")] = 1
            df.iloc[0, df.columns.get_loc("pHVACyn")] = "NO"
    else:
        # Baseline: the building's existing equipment. Heating and cooling are
        # installed independently: either, both, or none.
        heating_covers_cooling = False   # a heat pump also does the cooling
        if not heating_none:
            if heating_id in hvac_df["ty"].values:
                hvac_heating_row = hvac_df[hvac_df["ty"] == heating_id]
                heating_covers_cooling = (not hvac_heating_row.empty and
                                          float(hvac_heating_row["pHVmx"].iloc[0]) > 0 and
                                          float(hvac_heating_row["pACmx"].iloc[0]) > 0)
                if heating_id in df["pHVAC0"].values:
                    df.loc[df["pHVAC0"] == heating_id, "pHVACz0"] = 1
                else:
                    df.iloc[0, df.columns.get_loc("pHVAC0")] = heating_id
                    df.iloc[0, df.columns.get_loc("pHVACz0")] = 1
            else:
                if heating_id in df["pCHP0"].values:
                    df.loc[df["pCHP0"] == heating_id, "pCHPz0"] = 1
                else:
                    df.iloc[0, df.columns.get_loc("pCHP0")] = heating_id
                    df.iloc[0, df.columns.get_loc("pCHPz0")] = 1

        if not cooling_none and not cooling_ducted_hp and not heating_covers_cooling:
            if cooling_id in df["pHVAC0"].values:
                df.loc[df["pHVAC0"] == cooling_id, "pHVACz0"] = 1
            else:
                available = df[df["pHVACz0"] != 1].index
                if len(available) > 0:
                    df.iloc[available[0], df.columns.get_loc("pHVAC0")] = cooling_id
                    df.iloc[available[0], df.columns.get_loc("pHVACz0")] = 1
                else:
                    df.iloc[0, df.columns.get_loc("pHVAC0")] = cooling_id
                    df.iloc[0, df.columns.get_loc("pHVACz0")] = 1

    # --- Water heater ---
    df["pWHz0"] = 0
    df["pWHyn"] = "NO"
    if wh_id in df["pWH0"].values:
        df.loc[df["pWH0"] == wh_id, "pWHz0"] = 1
    else:
        df.iloc[0, df.columns.get_loc("pWH0")] = wh_id
        df.iloc[0, df.columns.get_loc("pWHz0")] = 1

    # --- Solar PV ---
    # When has_solar=True, fix PV as pre-existing equipment:
    # pPVz0 = panel_count (number of panels), pPVyn = "NO" (fixed).
    # Follows create_pv_sp_case.py from opendss_test.
    if has_solar and panel_count > 0:
        sp_pv_row = df["pPV0"] == PV_TECH
        if sp_pv_row.any():
            df.loc[sp_pv_row, "pPVz0"] = panel_count
            df.loc[sp_pv_row, "pPVyn"] = "NO"

    df = pd.concat([df, zeros_row], ignore_index=True)
    df.to_csv(str(output_dir / "sp.csv"), index=False)
    return df


def create_wh_file(output_dir, template, building_row=None):
    df = template.copy()
    if building_row is not None:
        GAL_TO_KWH = 3.78541 * 4.186 * 45 / 3600
        tank_gal = float(building_row["out.params.size_water_heater..gal"].iloc[0])
        tank_kwh = round(tank_gal * GAL_TO_KWH, 2)
        zeros_mask = df["ty"].astype(str).isin(["0", "0.0"])
        zeros_row = df[zeros_mask].copy()
        df = df[~zeros_mask].copy()
        tankless_mask = df["tank"].astype(float) == 0
        df.loc[~tankless_mask, "tank"] = round(tank_kwh)
        df = pd.concat([df, zeros_row], ignore_index=True)
    df.to_csv(str(output_dir / "wh.csv"), index=False)


def create_topo_file(output_dir, template, sp_df=None):
    df = template.copy()
    if sp_df is not None:
        installed = sp_df[
            (sp_df["pCHPz0"].astype(str) == "1") &
            (sp_df["pCHP0"].astype(str) != "0")
        ]["pCHP0"].tolist()
        zeros_mask = df["up"].astype(str).isin(["0", "0.0"])
        zeros_row = df[zeros_mask].copy()
        df = df[~zeros_mask].copy()
        existing = df["up"].tolist()
        for chp_name in installed:
            if chp_name not in existing:
                new_row = {
                    "up": chp_name, "lo": "Building", "in": "hot air", "eff": 1,
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = pd.concat([df, zeros_row], ignore_index=True)
    df.to_csv(str(output_dir / "topo.csv"), index=False)


# ---------------------------------------------------------------------------
# Worker state (per-process)
# ---------------------------------------------------------------------------

_worker_state = {}


def _worker_init(cfg, dcfg, metadata_df, regression_coeffs,
                 templates, elevation_cache, hp_summary_df, study_dir):
    from run_log import attach_run_log
    attach_run_log()
    _worker_state["cfg"] = cfg
    _worker_state["dcfg"] = dcfg
    _worker_state["year"] = cfg["year"]
    _worker_state["templates"] = templates
    _worker_state["elevation_cache"] = elevation_cache
    _worker_state["regression_coeffs"] = regression_coeffs
    _worker_state["study_dir"] = study_dir

    id_col = "bldg_id" if "bldg_id" in metadata_df.columns else "building_id"
    _worker_state["metadata_df"] = metadata_df.set_index(id_col)
    _worker_state["hp_summary_df"] = hp_summary_df.set_index("bldg_id")


# ---------------------------------------------------------------------------
# Per-building orchestrator
# ---------------------------------------------------------------------------

def process_building(args):
    bldg_id, tariff, elec_level, has_solar, case_name = args
    t0 = time.perf_counter()
    result = {"bldg_id": bldg_id, "case": case_name,
              "success": False, "elapsed": 0.0, "error": None}

    cfg = _worker_state["cfg"]
    dcfg = _worker_state["dcfg"]
    metadata_df = _worker_state["metadata_df"]
    regression_coeffs = _worker_state["regression_coeffs"]
    templates = _worker_state["templates"]
    elevation_cache = _worker_state["elevation_cache"]
    hp_summary_df = _worker_state["hp_summary_df"]
    input_dir = cfg["paths"]["input_dir"]
    state = cfg["state"]
    year = cfg["year"]
    outputs = dcfg["outputs"]

    try:
        building_row = get_building_row(bldg_id, metadata_df)

        consumption_df = load_consumption(bldg_id, input_dir, year, state)
        setpoint_df = None
        schedule_df = None
        try:
            setpoint_df = load_setpoints(bldg_id, input_dir, year, state)
        except FileNotFoundError:
            pass
        try:
            schedule_df = load_schedule(bldg_id, input_dir, state)
        except Exception:
            pass

        output_dir = _worker_state["study_dir"] / "decarb" / case_name / str(bldg_id) / "in"
        output_dir.mkdir(parents=True, exist_ok=True)

        if outputs["tm"] or outputs["cfg"]:     # cfg.csv pP0 follows tm.csv
            tm_df = generate_tm_data(
                consumption_df, setpoint_df, building_row, schedule_df,
                tariff, dcfg)
        if outputs["tm"]:
            create_tm_file(tm_df, output_dir)

        if outputs["bdg_ii"]:
            create_bdg_ii_file(output_dir, bldg_id, regression_coeffs)

        if outputs["cfg"]:
            create_cfg_file(output_dir, templates["cfg"], consumption_df, tm_df,
                            dcfg["timestep"])

        if outputs["other"]:
            create_other_files(output_dir, templates, has_solar=has_solar)

        chp_df = None
        hvac_df = None
        if outputs["chp"] or outputs["sp"]:
            chp_df = _build_chp_df(
                templates["chp"], bldg_id, hp_summary_df, building_row)
            if outputs["chp"]:
                chp_df.to_csv(str(output_dir / "chp.csv"), index=False)

        if outputs["hvac"] or outputs["sp"]:
            hvac_df = _build_hvac_df(
                templates["hvac"], bldg_id, hp_summary_df, building_row,
                setpoint_df=setpoint_df)
            if outputs["hvac"]:
                hvac_df.to_csv(str(output_dir / "hvac.csv"), index=False)

        if outputs["wh"]:
            create_wh_file(output_dir, templates["wh"], building_row)

        # --- Electrification: resolve heat pump name ---
        hp_name = None
        if elec_level > 0 and bldg_id in hp_summary_df.index:
            hp_row = hp_summary_df.loc[[bldg_id]]
            hp_index = str(hp_row["hp_index"].iloc[0])
            hp_name = f"Mitsubishi ASHP {hp_index}"

        # --- Solar: compute panel count and orientation ---
        panel_count = 0
        pv_roof_area = None
        pv_tilt = None
        pv_azimuth = None
        if has_solar:
            # Panel specs from pv.csv template
            pv_template = templates["pv.csv"]
            pv_row = pv_template[pv_template.index == PV_TECH]
            if not pv_row.empty:
                # Template read with header=None, index_col=0 → columns [1,2,3,...]
                # col 2 = mx (per-panel kW), col 3 = ar (per-panel m2)
                panel_rating_kw = float(pv_row[2].iloc[0])
                panel_area_m2 = float(pv_row[3].iloc[0])
            else:
                log.warning(f"Building {bldg_id}: {PV_TECH} not found in pv.csv "
                            f"template; using fallback panel 0.34 kW, 1.0 m2")
                panel_rating_kw = 0.34
                panel_area_m2 = 1.0
            panel_count = compute_pv_panel_count(consumption_df, panel_rating_kw)
            pv_roof_area = compute_required_pv_area(panel_count, panel_area_m2)

            # Compute optimal tilt/azimuth from tm.csv (already written)
            tm_path = output_dir / "tm.csv"
            if tm_path.exists():
                try:
                    tm_for_orient = pd.read_csv(str(tm_path), usecols=["pP", "pSdni"])
                    lat = float(building_row["in.weather_file_latitude"].iloc[0])
                    lon = float(building_row["in.weather_file_longitude"].iloc[0])
                    # Elevation is only looked up when bdg_i is on; otherwise 0 m
                    # (warned once in run_decarb_inputs)
                    alt = elevation_cache.get((lat, lon), 0.0)
                    pv_tilt, pv_azimuth = compute_optimal_orientation(
                        lat, lon, alt, dcfg["timestep"], tm_for_orient)
                except Exception as e:
                    log.warning(f"Building {bldg_id}: PV orientation failed "
                                f"({type(e).__name__}: {e}); using fallback "
                                f"tilt=35, azimuth=180")
                    pv_tilt, pv_azimuth = 35.0, 180.0

        sp_df = None
        if outputs["sp"]:
            sp_df = create_sp_file(
                output_dir, templates["sp"], building_row,
                hvac_df, chp_df, templates["wh"],
                elec_level=elec_level, has_solar=has_solar,
                hp_name=hp_name, panel_count=panel_count)

        if outputs["topo"]:
            create_topo_file(output_dir, templates["topo"], sp_df)

        if outputs["bdg_i"]:
            pBhvac = None
            if sp_df is not None:
                pBhvac = max(int((sp_df["pHVACz0"] == 1).sum()), 1)
            create_bdg_i_file(
                output_dir, templates["bdg_i"],
                building_row, elevation_cache, pBhvac=pBhvac,
                pv_roof_area=pv_roof_area, pv_tilt=pv_tilt,
                pv_azimuth=pv_azimuth)

        result["success"] = True

    except Exception:
        result["error"] = traceback.format_exc()

    result["elapsed"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _next_study_number(decarb_root):
    """Find the next available study number under decarb_root."""
    if not decarb_root.exists():
        return 0
    existing = [
        int(m.group(1))
        for d in decarb_root.iterdir()
        if d.is_dir()
        for m in [re.match(r"^study_(\d+)$", d.name)]
        if m
    ]
    return max(existing, default=-1) + 1


def _required_template_files(outputs):
    """Template files load_templates() reads for the enabled outputs."""
    files = []
    if outputs["bdg_i"]:
        files.append("bdg_i.csv")
    if outputs["cfg"]:
        files.append("cfg.csv")
    if outputs["other"]:
        files += OTHER_TEMPLATE_FILES
    if outputs["chp"] or outputs["sp"]:
        files.append("chp.csv")
    if outputs["topo"]:
        files.append("topo.csv")
    if outputs["hvac"] or outputs["sp"]:
        files.append("hvac.csv")
    if outputs["wh"] or outputs["sp"]:
        files.append("wh.csv")
    if outputs["sp"]:
        files.append("sp.csv")
    return files


def _check_required_files(cfg, dcfg, building_ids, metadata_df, hp_summary_df):
    """Stop before creating a study if any input decarb-inputs needs is missing.

    Global files raise immediately; per-building gaps are collected, written to
    out/decarb_inputs_missing_files.csv and reported together.
    """
    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    state = cfg["state"]
    outputs = dcfg["outputs"]

    template_dir = dcfg["template_dir"]
    missing_templates = [f for f in _required_template_files(outputs)
                         if not (template_dir / f).exists()]
    if missing_templates:
        raise FileNotFoundError(
            f"DECARB template file(s) missing from {template_dir}: "
            f"{', '.join(missing_templates)} (template_dir in decarb.yaml)")

    def _names(d):
        return set(os.listdir(d)) if d.is_dir() else set()

    profiles = _names(input_dir / "residential_profiles" / state)
    setpoints = _names(input_dir / "setpoint_timeseries" / state)
    schedules = _names(input_dir / "schedules" / state)
    id_col = "bldg_id" if "bldg_id" in metadata_df.columns else "building_id"
    meta_ids = set(metadata_df[id_col].astype(int))
    hp_ids = set(hp_summary_df["bldg_id"].astype(int))
    needs_hp = outputs["chp"] or outputs["hvac"] or outputs["sp"]
    needs_tm = outputs["tm"] or outputs["cfg"]   # cfg.csv is built from the tm.csv data

    checks = [
        ("ResStock profile (residential_profiles)", lambda b: f"{b}-0.parquet" in profiles),
        ("ResStock metadata row", lambda b: b in meta_ids),
    ]
    if needs_tm or needs_hp:   # hvac.csv uses the comfort band too
        checks.append(
            ("setpoint file (schedules stage)", lambda b: f"setpoint_{b}.parquet" in setpoints))
    if needs_tm:
        checks.append(
            ("schedule file (schedules stage)", lambda b: f"schedule_{b}.parquet" in schedules))
    if needs_hp:
        checks.append(("heat pump summary row", lambda b: b in hp_ids))

    missing = [{"bldg_id": b, "missing": what}
               for b in building_ids for what, ok in checks if not ok(b)]
    if not missing:
        log.info(f"All required inputs present for {len(building_ids)} buildings")
        return

    missing_df = pd.DataFrame(missing)
    report = output_dir / "decarb_inputs_missing_files.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    missing_df.to_csv(report, index=False)
    counts = "\n".join(f"  {n:>6}  {what}"
                       for what, n in missing_df["missing"].value_counts().items())
    raise FileNotFoundError(
        f"{missing_df['bldg_id'].nunique()} of {len(building_ids)} buildings are missing "
        f"inputs (full list: {report}):\n{counts}\n"
        f"Re-run the preprocessing stage that produces them, then try again.")


def run_decarb_inputs(cfg, dcfg=None, study_dir=None):
    """Generate DECARB input files for all cases defined in decarb.yaml.

    Parameters
    ----------
    cfg : dict
        Main pipeline config (from config.yaml).
    dcfg : dict, optional
        DECARB config (from decarb.yaml). Loaded automatically if None.
    study_dir : Path or str, optional
        Existing study directory to add cases into. If None, creates a new
        study_N directory. Existing cases inside study_dir are skipped.
    """
    if dcfg is None:
        dcfg = load_decarb_config()

    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    state = cfg["state"]
    year = cfg["year"]
    outputs = dcfg["outputs"]

    # Timestep lives in config.yaml; the per-building code reads dcfg["timestep"]
    timestep = resolve_timestep(cfg, dcfg)
    dcfg = dict(dcfg, timestep=timestep)

    # Check thermal model results before creating a study directory
    from fit_thermal_model import thermal_coeff_csv_path
    coeff_csv = thermal_coeff_csv_path(cfg)
    if outputs["bdg_ii"] and not coeff_csv.exists():
        raise FileNotFoundError(
            f"No thermal model results at {coeff_csv}. Run the thermal model stage "
            f"first (stages.thermal_model: true in config.yaml), or set "
            f"outputs.bdg_ii: false in decarb.yaml.")

    if outputs["bdg_ii"]:
        log.info(f"bdg_ii: thermal coefficients written as fitted, pBtime = "
                 f"{THERMAL_FIT_HOURS} h; DECARB converts them to {timestep} steps")

    tz = cfg.get("timezone")
    if not tz:
        raise KeyError("config.yaml is missing timezone (local clock for TOU peak "
                       "hours and DECARB output dates), e.g. timezone: \"America/Chicago\"")
    try:
        pd.Timestamp(f"{year}-01-01").tz_localize(tz)
    except Exception as e:
        raise ValueError(f"timezone {tz!r} in config.yaml is not a valid IANA "
                         f"time zone name: {e}") from e

    if study_dir is not None:
        study_dir = Path(study_dir)
        if not study_dir.exists():
            raise FileNotFoundError(f"Provided study_dir does not exist: {study_dir}")

    # ── Resolve and check every input before creating the study ──────────
    # Building IDs: the study's own matching file if adding to an existing
    # study that has one, else the matching file a new study would copy in
    matching_src = input_dir / "feeder_summaries_matched.parquet"
    matching_file = matching_src
    if study_dir is not None and (study_dir / "feeder_summaries_matched.parquet").exists():
        matching_file = study_dir / "feeder_summaries_matched.parquet"
    if matching_file.exists():
        matched_df = read_parquet(matching_file)
        if cfg.get("scope") == "zone" and "zone" in matched_df.columns:
            file_zones = sorted(matched_df["zone"].dropna().unique())
            zone_name = cfg["zone"]["name"]
            if file_zones != [zone_name]:
                fix = ("Re-run building matching (stages.building_matching: true)"
                       if matching_file == matching_src else
                       "Use a study made for this zone, or change zone.name")
                raise ValueError(
                    f"{matching_file} is for zone(s) {', '.join(file_zones) or 'none'}, "
                    f"but zone.name in config.yaml is {zone_name!r}. {fix}.")
        bldg_col = "nrel_climate_match" if "nrel_climate_match" in matched_df.columns else "nrel_county_match"
        building_ids = (matched_df[bldg_col].dropna().astype(int)
                        .unique().tolist())
        building_ids.sort()
        log.info(f"Building IDs from matching: {len(building_ids)} unique buildings")
    else:
        # Fallback: raw ID file (no matching available yet)
        from data_acquisition import load_building_ids
        building_ids = load_building_ids(cfg)
        log.info(f"Building IDs from ID file: {len(building_ids)} "
                 f"(no matching file in study)")

    # Load metadata
    meta_path = input_dir / f"{state}_upgrade0.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"ResStock metadata not found: {meta_path}. Run the data acquisition "
            f"stage first (stages.data_acquisition: true in config.yaml).")
    metadata_df = read_parquet(meta_path)

    # Load regression coefficients from the thermal model stage
    regression_coeffs = {}
    if coeff_csv.exists():
        coeff_df = pd.read_csv(str(coeff_csv))
        for _, row in coeff_df.iterrows():
            regression_coeffs[str(int(row["bldg_id"]))] = {
                "k1": float(row["k1"]),
                "k2": float(row["k2"]),
                "k3": float(row["k3"]),
            }
        log.info(f"Loaded regression coefficients for {len(regression_coeffs)} buildings "
                 f"from {coeff_csv}")
        if not regression_coeffs and outputs["bdg_ii"]:
            raise ValueError(
                f"Thermal model results at {coeff_csv} contain no buildings. "
                f"Check the thermal model stage log for why no fits succeeded.")

    if regression_coeffs:
        fitted_ids = set(int(k) for k in regression_coeffs.keys())
        before = len(building_ids)
        building_ids = [b for b in building_ids if b in fitted_ids]
        if len(building_ids) < before:
            log.info(f"Filtered to {len(building_ids)}/{before} buildings "
                     f"with thermal model results")
    else:
        # Only reachable with outputs.bdg_ii: false (checked above)
        log.warning("No thermal model results; bdg_ii is off and the building "
                    "list is not filtered to fitted buildings")

    # Load HP summary
    hp_summary_path = output_dir / "heatpump_files" / state / "building_hp_summary.csv"
    if not hp_summary_path.exists():
        raise FileNotFoundError(
            f"Heat pump summary not found: {hp_summary_path}. Run the heat pump "
            f"selection stage first (stages.heatpump_selection: true in config.yaml).")
    hp_summary_df = pd.read_csv(str(hp_summary_path))

    _check_required_files(cfg, dcfg, building_ids, metadata_df, hp_summary_df)

    # Load templates
    templates = load_templates(dcfg["template_dir"], outputs)

    # Build elevation cache if needed
    elevation_cache = {}
    if outputs["bdg_i"]:
        lat_col = "in.weather_file_latitude"
        lon_col = "in.weather_file_longitude"
        id_col = "bldg_id" if "bldg_id" in metadata_df.columns else "building_id"
        subset = metadata_df[metadata_df[id_col].isin(building_ids)]
        # float keys, matching the float(...) lookups in create_bdg_i_file
        unique_coords = list(
            subset[[lat_col, lon_col]].astype(float).drop_duplicates().itertuples(
                index=False, name=None))
        elevation_cache = build_elevation_cache(unique_coords, max_workers=20)
    elif dcfg.get("rooftop_solar", {}).get("enabled", False):
        log.warning("outputs.bdg_i is off, so no elevations are looked up; PV "
                    "orientation will use 0 m elevation for every building")

    # ── All inputs are in place: create or reuse the study ───────────────
    if study_dir is not None:
        log.info(f"Adding cases to existing study at {study_dir}")
    else:
        # Create study directory with next available number
        studies_root = output_dir / "studies"
        studies_root.mkdir(parents=True, exist_ok=True)
        study_num = _next_study_number(studies_root)
        study_dir = studies_root / f"study_{study_num}"
        study_dir.mkdir()
        log.info(f"Created study_{study_num} at {study_dir}")

        # Prompt user for study description and log it
        description = input(f"Enter a description for study_{study_num}: ").strip()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        study_log = studies_root / "study_log.csv"
        write_header = not study_log.exists()
        with open(study_log, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["study_id", "timestamp", "description"])
            writer.writerow([f"study_{study_num}", timestamp, description])

    # Save copies of the config files that generated this study
    config_dir = Path(__file__).parents[1] / "config"
    for cfg_file in ("config.yaml", "decarb.yaml", "network.yaml"):
        src = config_dir / cfg_file
        if src.exists():
            shutil.copy2(str(src), str(study_dir / cfg_file))
            log.info(f"Saved {cfg_file} to study directory")
        else:
            log.warning(f"Config file not found: {src}")

    # Save the matching file into the study (ties buildings to this study)
    matching_dst = study_dir / "feeder_summaries_matched.parquet"
    if matching_src.exists() and not matching_dst.exists():
        shutil.copy2(str(matching_src), str(matching_dst))
        log.info(f"Saved matching file to study: {matching_dst.name}")

    # Build case list
    elec_levels = dcfg["electrification"]["levels"]
    tariffs = dcfg["tariffs"]
    solar_cfg = dcfg.get("rooftop_solar", {})
    solar_enabled = solar_cfg.get("enabled", False)

    # Build bookend cases: tariff × electrification × solar (0% and 100%)
    # All buildings get solar parameters calculated, but mx is only
    # nonzero in the solar=100 case. Intermediate penetration levels
    # are handled at the network demand stage.
    cases = []
    for elec in elec_levels:
        for tariff in tariffs:
            # Always generate the no-solar case
            case_name = f"{tariff['name']}_heat_{elec}_solar_0"
            cases.append((tariff, elec, False, case_name))

            # Generate the all-solar case if solar is enabled
            if solar_enabled:
                case_name = f"{tariff['name']}_heat_{elec}_solar_100"
                cases.append((tariff, elec, True, case_name))

    max_workers = dcfg.get("max_workers", 4)

    for tariff, elec_level, has_solar, case_name in cases:
        # Skip cases that already exist in the study directory
        case_path = study_dir / "decarb" / case_name
        if case_path.exists():
            log.info(f"Skipping existing case: {case_name}")
            continue

        log.info(f"Generating case: {case_name} ({len(building_ids)} buildings"
                 + (", all with solar)" if has_solar else ")"))

        work_items = [
            (bldg_id, tariff, elec_level, has_solar, case_name)
            for bldg_id in building_ids
        ]

        successes = 0
        failures = 0

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(cfg, dcfg, metadata_df, regression_coeffs,
                      templates, elevation_cache, hp_summary_df, study_dir),
        ) as executor:
            futures = {
                executor.submit(process_building, item): item
                for item in work_items
            }
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result["success"]:
                    successes += 1
                else:
                    failures += 1
                    if result["error"]:
                        err_lines = result["error"].strip().split("\n")
                        log.warning(
                            f"  Failed bldg {result['bldg_id']}: {err_lines[-1]}")

                if i % 100 == 0 or i == len(work_items):
                    log.info(f"  [{i}/{len(work_items)}] "
                             f"ok={successes} fail={failures}")

        log.info(f"Case {case_name} done: {successes} ok, {failures} failed")
        log.info(f"Output: {study_dir / 'decarb' / case_name}")

    # Zip the decarb folder for easy transfer
    decarb_dir = study_dir / "decarb"
    if decarb_dir.exists():
        zip_path = study_dir / "decarb.zip"
        log.info(f"Zipping decarb folder → {zip_path.name} ...")
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", study_dir, "decarb")
        log.info(f"Saved {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    return study_dir


if __name__ == "__main__":
    import sys
    cfg = load_config()
    dcfg = load_decarb_config()
    # Optional: pass existing study directory as command-line argument
    # to add new cases without creating a new study
    existing_study = sys.argv[1] if len(sys.argv) > 1 else None
    run_decarb_inputs(cfg, dcfg, study_dir=existing_study)
