import re
import logging
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).parents[1]))
from config.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ZONE_PATTERN = re.compile(r"^P\d+[RU]$")


def load_county_shapefile(shapefile_path):
    if not Path(shapefile_path).exists():
        raise FileNotFoundError(
            f"County shapefile not found: {shapefile_path}. It ships with the repo in "
            f"in/us_county_boundaries/; check paths.county_shapefile in config.yaml "
            f"(see README, Input data).")
    counties = gpd.read_file(str(shapefile_path))
    counties = counties.to_crs(epsg=4326)
    log.info(f"Loaded {len(counties)} counties from shapefile")
    return counties


def get_county_from_coordinates(lat, lon, counties):
    if pd.isna(lat) or pd.isna(lon):
        return "Not found", None

    point = Point(lon, lat)
    possible = counties[counties.geometry.intersects(point)]

    if not possible.empty:
        county = possible.iloc[0]
        name = county["name"]
        code = f"G{county['statefp']}0{county['countyfp']}0"
        return name, code

    return "Not found", None


def find_zones(network_dir):
    zones = []
    for d in sorted(Path(network_dir).iterdir()):
        if d.is_dir() and ZONE_PATTERN.match(d.name):
            zones.append(d)
    return zones


TEST_MODE_STATEWIDE_ZONES = 3


def select_zones(cfg, zone_dirs):
    """Zones to work on: only zone.name for scope: zone (full and test runs);
    every zone for scope: statewide, or the first TEST_MODE_STATEWIDE_ZONES
    in test mode."""
    if cfg.get("scope") == "zone":
        zone_name = cfg["zone"]["name"]
        selected = [d for d in zone_dirs if d.name == zone_name]
        if not selected:
            raise ValueError(
                f"Zone {zone_name!r} (zone.name in config.yaml) not found in "
                f"{cfg['paths']['network_dir']}")
        return selected

    if cfg.get("test_mode"):
        selected = zone_dirs[:TEST_MODE_STATEWIDE_ZONES]
        log.warning(f"TEST MODE: limited to zone(s) {', '.join(d.name for d in selected)}")
        return selected
    return zone_dirs


def list_feeders(opendss_dir):
    """Feeder folders of a SMART-DS zone: {opendss}/{substation}/{feeder}."""
    feeders = []
    for sub in sorted(Path(opendss_dir).iterdir()):
        if not sub.is_dir() or sub.name == "opendss_loadshape_files":
            continue
        feeders += [fd for fd in sorted(sub.iterdir())
                    if fd.is_dir() and fd.name != "feeder_summaries"]
    return feeders


def feeder_selection(cfg):
    """zone.substations and zone.feeders from config.yaml (None = not set)."""
    zone = cfg.get("zone") or {}
    subs, feeds = zone.get("substations"), zone.get("feeders")
    if (subs or feeds) and cfg.get("scope") != "zone":
        raise ValueError("zone.substations / zone.feeders only apply to scope: \"zone\"")
    return (list(subs) if subs else None), (list(feeds) if feeds else None)


def select_feeders(cfg, feeder_dirs):
    """Feeders to work on: all of them, or those under zone.substations plus
    those named in zone.feeders. Unknown names are an error."""
    subs, feeds = feeder_selection(cfg)
    if not subs and not feeds:
        return feeder_dirs
    all_subs = sorted({fd.parent.name for fd in feeder_dirs})
    all_feeds = sorted(fd.name for fd in feeder_dirs)
    for key, given, valid in (("substations", subs, all_subs), ("feeders", feeds, all_feeds)):
        unknown = sorted(set(given or []) - set(valid))
        if unknown:
            raise ValueError(
                f"zone.{key} in config.yaml: {', '.join(unknown)} not in zone "
                f"{cfg['zone']['name']}. Available: {', '.join(valid[:20])}"
                + (f", ... ({len(valid)} total)" if len(valid) > 20 else ""))
    selected = [fd for fd in feeder_dirs
                if (subs and fd.parent.name in subs) or (feeds and fd.name in feeds)]
    log.info(f"Feeder selection: {len(selected)} of {len(feeder_dirs)} feeders "
             f"(zone.substations={subs}, zone.feeders={feeds})")
    return selected


def check_matching_feeders(cfg, matched_df, label):
    """Stop if a matching file covers feeders outside the current selection
    (it was made with a different zone.substations / zone.feeders); warn about
    selected feeders that have residential loads but none in the file."""
    if cfg.get("scope") != "zone" or not {"substation", "feeder"} <= set(matched_df.columns):
        return
    opendss_dir = (Path(cfg["paths"]["network_dir"]) / cfg["zone"]["name"]
                   / "scenarios" / "base_timeseries" / "opendss")
    selected = select_feeders(cfg, list_feeders(opendss_dir))
    sel = {(fd.parent.name, fd.name) for fd in selected}
    in_file = set(zip(matched_df["substation"], matched_df["feeder"]))
    extra = sorted(in_file - sel)
    if extra:
        raise ValueError(
            f"{label} includes {len(extra)} feeder(s) outside the current feeder selection "
            f"(zone.substations / zone.feeders), e.g. {extra[0][1]}. Re-run feeder processing "
            f"and building matching, or change the selection back.")
    subs, feeds = feeder_selection(cfg)
    if subs or feeds:
        missing = [fd.name for fd in selected if (fd.parent.name, fd.name) not in in_file
                   and (fd / "Loads.dss").exists()
                   and re.search(r"(?i)yearly\s*=\s*res", (fd / "Loads.dss").read_text())]
        if missing:
            log.warning(f"{label} has no loads for {len(missing)} selected feeder(s) with "
                        f"residential loads, e.g. {missing[0]}; re-run feeder processing and "
                        f"building matching if the selection changed")


DSS_NEW_OBJECT = re.compile(r"^new\s+(\w+)\.(\S+)\s*(.*)$", re.IGNORECASE)
DSS_PROPERTY = re.compile(r"(\w+)\s*=\s*(\([^)]*\)|\[[^\]]*\]|\"[^\"]*\"|'[^']*'|[^\s]+)")


def _strip_dss_comment(line):
    for marker in ("!", "//"):
        line = line.split(marker, 1)[0]
    return line.strip()


def parse_dss_objects(dss_path, object_class):
    """Read 'New <Class>.<name> key=value ...' definitions from a .dss file.

    Returns a DataFrame with a 'Name' column (without the class prefix) and
    one column per property, keys lowercased (kV -> kv, kW -> kw)."""
    rows = []
    for raw in Path(dss_path).read_text().splitlines():
        line = _strip_dss_comment(raw)
        if not line:
            continue
        if line.startswith("~") and rows:            # continuation of previous object
            props_text = line[1:]
        else:
            m = DSS_NEW_OBJECT.match(line)
            if not m or m.group(1).lower() != object_class.lower():
                continue
            rows.append({"Name": m.group(2)})
            props_text = m.group(3)
        for key, value in DSS_PROPERTY.findall(props_text):
            rows[-1][key.lower()] = value.strip("\"'")

    df = pd.DataFrame(rows)
    for col in df.columns:
        if col != "Name":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() == df[col].notna().sum():
                df[col] = converted
    return df


def parse_buscoords(dss_path):
    """Read 'bus x y' lines (x = longitude, y = latitude) from Buscoords.dss."""
    rows = []
    for raw in Path(dss_path).read_text().splitlines():
        parts = _strip_dss_comment(raw).replace(",", " ").split()
        if len(parts) >= 3:
            rows.append((parts[0], float(parts[1]), float(parts[2])))
    return pd.DataFrame(rows, columns=["Bus_Name", "Longitude", "Latitude"])


def read_feeder_loads(feeder_path):
    """Loads of one feeder with their bus coordinates, read from the SMART-DS
    Loads.dss and Buscoords.dss. Empty DataFrame if the feeder has neither."""
    loads_file = feeder_path / "Loads.dss"
    buscoords_file = feeder_path / "Buscoords.dss"

    if not loads_file.exists() or not buscoords_file.exists():
        return pd.DataFrame()

    loads_df = parse_dss_objects(loads_file, "Load")
    if loads_df.empty:
        return loads_df
    buscoords_df = parse_buscoords(buscoords_file)

    loads_df["bus1_base"] = loads_df["bus1"].apply(lambda x: str(x).split(".")[0])
    merged_df = pd.merge(
        loads_df,
        buscoords_df,
        left_on=loads_df["bus1_base"].str.lower(),
        right_on=buscoords_df["Bus_Name"].str.lower(),
        how="left",
    ).drop(columns=["key_0"], errors="ignore")
    return merged_df


def process_feeder(feeder_path, feeder_name, substation_name, zone_name, counties):
    merged_df = read_feeder_loads(feeder_path)
    if merged_df.empty:
        return merged_df

    merged_df[["county", "county_code"]] = merged_df.apply(
        lambda row: pd.Series(
            get_county_from_coordinates(row["Latitude"], row["Longitude"], counties)
        ),
        axis=1,
    )

    merged_df["zone"] = zone_name
    merged_df["substation"] = substation_name
    merged_df["feeder"] = feeder_name

    merged_df.drop(columns=["Bus_Name", "bus1_base"], inplace=True, errors="ignore")

    return merged_df


def process_zone(zone_dir, counties, cfg):
    zone_name = zone_dir.name
    opendss_path = zone_dir / "scenarios" / "base_timeseries" / "opendss"

    if not opendss_path.exists():
        log.warning(f"No opendss path for {zone_name}")
        return pd.DataFrame()

    feeders = select_feeders(cfg, list_feeders(opendss_path))
    substations = sorted({fd.parent.name for fd in feeders})

    if not substations:
        log.warning(f"No substations found in {zone_name}")
        return pd.DataFrame()

    zone_records = []

    for feeder_dir in feeders:
        df = process_feeder(
            feeder_dir, feeder_dir.name, feeder_dir.parent.name, zone_name, counties
        )
        if not df.empty:
            zone_records.append(df)

    if not zone_records:
        return pd.DataFrame()

    zone_df = pd.concat(zone_records, ignore_index=True)

    residential_mask = zone_df["yearly"].astype(str).str.startswith("res", na=False)
    zone_res = zone_df[residential_mask].copy()

    log.info(f"{zone_name}: {len(zone_df)} total loads, {len(zone_res)} residential, "
             f"{len(substations)} substations")

    return zone_res


def build_county_to_climate_lookup(cfg):
    input_dir = cfg["paths"]["input_dir"]
    state = cfg["state"]
    metadata_path = input_dir / f"{state}_upgrade0.parquet"

    try:
        meta = pd.read_parquet(str(metadata_path), engine="fastparquet")
    except Exception:
        meta = pd.read_parquet(str(metadata_path), engine="pyarrow")

    county_col = "in.county"
    climate_col = "in.building_america_climate_zone"

    lookup = (
        meta[[county_col, climate_col]]
        .dropna()
        .drop_duplicates()
        .set_index(county_col)[climate_col]
        .to_dict()
    )
    log.info(f"Built county→climate lookup: {len(lookup)} counties")
    return lookup


def process_all_zones(cfg):
    network_dir = cfg["paths"]["network_dir"]
    input_dir = cfg["paths"]["input_dir"]
    shapefile_path = cfg["paths"]["county_shapefile"]

    counties = load_county_shapefile(shapefile_path)
    county_to_climate = build_county_to_climate_lookup(cfg)

    zones = find_zones(network_dir)
    if not zones:
        raise FileNotFoundError(
            f"No SMART-DS zone folders found in {network_dir} (paths.network_dir). "
            f"Turn on data_acquisition.download_network and run data acquisition.")

    zones = select_zones(cfg, zones)

    log.info(f"Processing {len(zones)} zone(s) from {network_dir}: "
             f"{', '.join(z.name for z in zones[:10])}{' ...' if len(zones) > 10 else ''}")

    all_dfs = []
    for i, zone_dir in enumerate(zones, 1):
        zone_df = process_zone(zone_dir, counties, cfg)
        if not zone_df.empty:
            all_dfs.append(zone_df)
        log.info(f"  [{i}/{len(zones)}] {zone_dir.name} done")

    if not all_dfs:
        raise ValueError(
            f"No residential loads found in {', '.join(z.name for z in zones)}: no feeder "
            f"folder had both Loads.dss and Buscoords.dss with 'res' loads. Check "
            f"{network_dir}, or re-download the network.")

    combined = pd.concat(all_dfs, ignore_index=True)

    combined["climate_zone"] = combined["county_code"].map(county_to_climate)
    unmapped = combined["climate_zone"].isna().sum()
    if unmapped > 0:
        log.warning(f"{unmapped} buses could not be mapped to a climate zone")

    output_path = input_dir / "feeder_summaries.parquet"
    combined.to_parquet(str(output_path), index=False)
    log.info(f"Saved {len(combined)} rows ({combined['zone'].nunique()} zones) to {output_path}")

    summary = combined.groupby("zone").agg(
        total_residential_buses=("Name", "count"),
        unique_counties=("county_code", "nunique"),
    ).reset_index()
    summary_path = input_dir / "feeder_summaries_by_zone.csv"
    summary.to_csv(str(summary_path), index=False)
    log.info(f"Zone summary saved to {summary_path}")

    return combined


if __name__ == "__main__":
    cfg = load_config()
    process_all_zones(cfg)
