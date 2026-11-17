import logging
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def download_workers(cfg):
    """Number of parallel downloads (data_acquisition.download_workers)."""
    n = (cfg.get("data_acquisition") or {}).get("download_workers")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"data_acquisition.download_workers in config.yaml must be "
                         f"an integer >= 1, got {n!r}")
    return n


def _s3_prefix(cfg):
    rs = cfg["resstock"]
    state = cfg["state"]
    return (
        f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        f"{rs['release_year']}/{rs['release']}/timeseries_individual_buildings/"
        f"by_state/upgrade=0/state={state}/"
    )


def _s3_client(max_connections=10):
    return boto3.client("s3", config=Config(signature_version=UNSIGNED,
                                            max_pool_connections=max_connections))


def building_id_file(cfg):
    """Full list of ResStock building IDs for the state (written by acquisition)."""
    return (cfg["paths"]["input_dir"]
            / f"resstock_bldg_ids_{cfg['state'].lower()}_{cfg['resstock']['release_year']}.csv")


def apply_test_limit(cfg, building_ids):
    """In test mode, keep the test_limit lowest building IDs; otherwise all."""
    building_ids = list(building_ids)
    if cfg.get("test_mode") and cfg.get("test_limit"):
        limited = sorted(building_ids)[: cfg["test_limit"]]
        log.warning(f"TEST MODE: using {len(limited)} of {len(building_ids):,} buildings")
        return limited
    return building_ids


def load_building_ids(cfg):
    """Building IDs every preprocessing stage runs on (test_limit applied)."""
    id_file = building_id_file(cfg)
    if not id_file.exists():
        raise FileNotFoundError(
            f"Missing building ID list: {id_file}. Run the data acquisition stage "
            f"first (stages.data_acquisition: true in config.yaml).")
    building_ids = pd.read_csv(id_file)["bldg_id"].dropna().astype(int).drop_duplicates()
    return apply_test_limit(cfg, building_ids)


def get_valid_building_ids(cfg):
    s3 = _s3_client()
    bucket = cfg["resstock"]["s3_bucket"]
    prefix = _s3_prefix(cfg)
    pattern = re.compile(r"(\d+)-\d+\.parquet")

    log.info(f"Listing building IDs from s3://{bucket}/{prefix}")

    building_ids = []
    continuation_token = None

    while True:
        params = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**params)

        for obj in response.get("Contents", []):
            match = pattern.search(obj["Key"])
            if match:
                building_ids.append(int(match.group(1)))

        continuation_token = response.get("NextContinuationToken")
        if not continuation_token:
            break

    log.info(f"Found {len(building_ids)} building IDs")

    # Always save the full list; test_limit is applied when the list is read
    id_file = building_id_file(cfg)
    id_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bldg_id": building_ids}).to_csv(id_file, index=False)
    log.info(f"Saved building ID list to {id_file}")

    return building_ids


def download_building_files(cfg, building_ids=None):
    if building_ids is None:
        building_ids = load_building_ids(cfg)
    else:
        building_ids = apply_test_limit(cfg, building_ids)

    s3 = _s3_client(download_workers(cfg))
    bucket = cfg["resstock"]["s3_bucket"]
    prefix = _s3_prefix(cfg)
    state = cfg["state"]
    output_dir = cfg["paths"]["input_dir"] / "residential_profiles" / state
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_file = output_dir / "failed_downloads.csv"

    to_download = [b for b in building_ids
                   if not (output_dir / f"{b}-0.parquet").exists()]
    skipped = len(building_ids) - len(to_download)
    workers = download_workers(cfg)
    log.info(f"Downloading {len(to_download)} buildings to {output_dir} "
             f"({skipped} already present, {workers} parallel downloads)")

    def _fetch(bldg_id):
        # download_file writes to a temporary name and renames it when
        # complete, so an interrupted download never leaves a partial file
        s3.download_file(bucket, f"{prefix}{bldg_id}-0.parquet",
                         str(output_dir / f"{bldg_id}-0.parquet"))

    downloaded = 0
    failed = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch, b): b for b in to_download}
        for future in as_completed(futures):
            bldg_id = futures[future]
            try:
                future.result()
                downloaded += 1
            except Exception as e:
                log.warning(f"Failed: {bldg_id} ({e})")
                failed.append((bldg_id, str(e)))
            done = downloaded + len(failed)
            if done % 500 == 0:
                rate = done / (time.time() - t_start)
                log.info(f"  {done}/{len(to_download)} ({rate:.1f}/s, "
                         f"~{(len(to_download) - done) / rate / 60:.0f} min left)")

    log.info(f"Done. downloaded={downloaded}, skipped={skipped}, failed={len(failed)} "
             f"({time.time() - t_start:.0f}s)")

    if failed:
        pd.DataFrame(failed, columns=["bldg_id", "error"]).to_csv(failed_file, index=False)
        raise RuntimeError(
            f"{len(failed)} building profile(s) failed to download (list: {failed_file}). "
            f"Re-run data acquisition to retry; finished files are skipped.")
    failed_file.unlink(missing_ok=True)

    return {"downloaded": downloaded, "skipped": skipped, "failed": []}


def _metadata_s3_key(cfg, fmt):
    rs = cfg["resstock"]
    state = cfg["state"]
    ext = "csv" if fmt == "csv" else "parquet"
    filename = f"{state}_upgrade0.{ext}"
    if fmt == "csv":
        filename += ".gz"
    return (
        f"nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        f"{rs['release_year']}/{rs['release']}/metadata_and_annual_results/"
        f"by_state/full/{ext}/state={state}/{filename}"
    )


def download_metadata(cfg):
    s3 = _s3_client()
    bucket = cfg["resstock"]["s3_bucket"]
    state = cfg["state"]
    output_dir = cfg["paths"]["input_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # The parquet is what every pipeline stage reads; the csv.gz is only
    # downloaded for visually checking the metadata by hand.
    files = {
        "csv": f"{state}_upgrade0.csv.gz",
        "parquet": f"{state}_upgrade0.parquet",
    }

    for fmt, filename in files.items():
        local_path = output_dir / filename
        if local_path.exists():
            log.info(f"Metadata {filename} already exists, skipping")
            continue

        key = _metadata_s3_key(cfg, fmt)
        log.info(f"Downloading metadata: s3://{bucket}/{key}")
        try:
            s3.download_file(bucket, key, str(local_path))
            log.info(f"Saved to {local_path}")
        except Exception as e:
            if fmt == "parquet":
                rs = cfg["resstock"]
                raise RuntimeError(
                    f"Failed to download ResStock metadata s3://{bucket}/{key}: {e}\n"
                    f"Check state ({state!r}), resstock.release ({rs['release']!r}) and "
                    f"resstock.release_year ({rs['release_year']!r}) in config.yaml.") from e
            log.warning(f"Failed to download {filename} (only used for manual checks): {e}")


def _collect_profile_filenames(feeder_dirs):
    """Profile CSVs referenced by the feeders' non-residential load shapes.
    Residential loads are replaced by DECARB results, so their profiles are
    not needed."""
    filenames = set()
    for path in (Path(fd) / "LoadShapes.dss" for fd in feeder_dirs):
        if not path.exists():
            continue
        for match in re.finditer(r'file=[^\s\)]*?profiles/([^\s\)]+)', path.read_text()):
            if not match.group(1).startswith("res_"):
                filenames.add(match.group(1))
    return filenames


def smartds_profiles_dir(cfg):
    """Where the SMART-DS profiles are stored: paths.network_dir/profiles, the
    bucket layout, so the feeders' relative profile paths resolve."""
    return Path(cfg["paths"]["network_dir"]) / "profiles"


def _smartds_location(cfg):
    sm = cfg.get("smartds") or {}
    missing = [k for k in ("s3_bucket", "s3_prefix") if not sm.get(k)]
    if missing:
        raise KeyError(f"config.yaml is missing smartds.{', smartds.'.join(missing)}")
    return sm["s3_bucket"], sm["s3_prefix"].strip("/")


def network_profiles_enabled(cfg):
    acq = cfg.get("data_acquisition") or {}
    if "download_network_profiles" in acq:
        return bool(acq["download_network_profiles"])
    matching = cfg.get("matching") or {}
    for old_key in ("download_network_profiles", "download_kvar"):
        if old_key in matching:
            log.warning(f"config key 'matching.{old_key}' is deprecated; move it to "
                        f"'data_acquisition.download_network_profiles'")
            return bool(matching[old_key])
    return False


def network_download_enabled(cfg):
    return bool((cfg.get("data_acquisition") or {}).get("download_network", False))


def _list_smartds_zones(s3, bucket, prefix):
    from process_network_info import ZONE_PATTERN
    zones, token = [], None
    while True:
        params = {"Bucket": bucket, "Prefix": f"{prefix}/", "Delimiter": "/"}
        if token:
            params["ContinuationToken"] = token
        resp = s3.list_objects_v2(**params)
        for p in resp.get("CommonPrefixes", []):
            name = p["Prefix"].rstrip("/").split("/")[-1]
            if ZONE_PATTERN.match(name):
                zones.append(name)
        token = resp.get("NextContinuationToken")
        if not token:
            return sorted(zones)


def _list_keys(s3, bucket, prefix):
    keys, token = [], None
    while True:
        params = {"Bucket": bucket, "Prefix": prefix}
        if token:
            params["ContinuationToken"] = token
        resp = s3.list_objects_v2(**params)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        token = resp.get("NextContinuationToken")
        if not token:
            return keys


def download_network(cfg):
    """Download the SMART-DS feeder .dss files (the zone's opendss/ folder)
    into paths.network_dir, keeping the bucket layout. Existing files are
    skipped. scope: zone -> zone.name only; statewide -> every zone."""
    from process_network_info import select_zones

    s3 = _s3_client(download_workers(cfg))
    bucket, prefix = _smartds_location(cfg)
    network_dir = Path(cfg["paths"]["network_dir"])

    available = _list_smartds_zones(s3, bucket, prefix)
    if not available:
        raise RuntimeError(f"No SMART-DS zones found at s3://{bucket}/{prefix}/ "
                           f"(check smartds in config.yaml)")

    if cfg.get("scope") == "zone":
        zone_name = cfg["zone"]["name"]
        if zone_name not in available:
            raise ValueError(
                f"Zone {zone_name!r} (zone.name in config.yaml) is not in SMART-DS at "
                f"s3://{bucket}/{prefix}/. Available: {', '.join(available[:20])}"
                + (f", ... ({len(available)} total)" if len(available) > 20 else ""))
        zones = [zone_name]
    else:
        zones = [d.name for d in select_zones(cfg, [network_dir / z for z in available])]
        if len(zones) > 10:
            log.warning(f"Downloading the network for {len(zones)} zones "
                        f"(roughly 25 MB each)")

    failed = []
    for zone in zones:
        zone_prefix = f"{prefix}/{zone}/scenarios/base_timeseries/opendss/"
        keys = [k for k in _list_keys(s3, bucket, zone_prefix) if not k.endswith("/")]
        if not keys:
            raise RuntimeError(f"No network files found at s3://{bucket}/{zone_prefix}")

        def _fetch(key):
            # download_file writes to a temporary name and renames it when
            # complete, so an interrupted download never leaves a partial file
            local = network_dir / key[len(prefix) + 1:]
            if local.exists():
                return "skipped"
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(bucket, key, str(local))
                return "downloaded"
            except Exception as e:
                return f"failed: {e}"

        counts = {"downloaded": 0, "skipped": 0}
        with ThreadPoolExecutor(max_workers=download_workers(cfg)) as executor:
            futures = {executor.submit(_fetch, k): k for k in keys}
            for future in as_completed(futures):
                result = future.result()
                if result in counts:
                    counts[result] += 1
                else:
                    failed.append((futures[future], result))
        log.info(f"Network {zone}: {counts['downloaded']} downloaded, "
                 f"{counts['skipped']} already present, "
                 f"{len(keys) - sum(counts.values())} failed "
                 f"-> {network_dir / zone}")

    if failed:
        examples = "\n".join(f"  {k}: {r}" for k, r in failed[:10])
        hint = ""
        longest = max(len(str(network_dir / k[len(prefix) + 1:])) for k, _ in failed)
        if os.name == "nt" and longest > 220:
            hint = (f"\nThe local paths are up to {longest} characters; Windows limits paths "
                    f"to 260 by default. Use a shorter paths.network_dir (e.g. D:/smartds) "
                    f"or enable Windows long path support.")
        raise RuntimeError(f"{len(failed)} network file(s) failed to download:\n{examples}\n"
                           f"Re-run data acquisition to retry; finished files are skipped.{hint}")


def download_network_profiles(cfg):
    """Download the SMART-DS kW and kvar profiles of the non-residential
    (commercial) loads on the selected feeders (zone.substations / zone.feeders,
    default: the whole zone) into smartds_profiles_dir(cfg). The OpenDSS cases
    keep these loads with their SMART-DS profiles. Existing files are skipped."""
    from process_network_info import list_feeders, select_feeders, select_zones

    s3 = _s3_client(download_workers(cfg))
    bucket, prefix = _smartds_location(cfg)
    network_dir = Path(cfg["paths"]["network_dir"])
    profiles_dir = smartds_profiles_dir(cfg)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    zone_dirs = select_zones(cfg, sorted(d for d in network_dir.iterdir()
                                         if d.is_dir() and d != profiles_dir))
    failed = []
    for zone_dir in zone_dirs:
        opendss_dir = zone_dir / "scenarios" / "base_timeseries" / "opendss"
        if not opendss_dir.is_dir():
            raise FileNotFoundError(
                f"No SMART-DS network for {zone_dir.name} at {opendss_dir}; turn on "
                f"data_acquisition.download_network and run data acquisition")
        filenames = sorted(_collect_profile_filenames(select_feeders(cfg, list_feeders(opendss_dir))))

        def _fetch(fn):
            local = profiles_dir / fn
            if local.exists():
                return "skipped"
            try:
                s3.download_file(bucket, f"{prefix}/profiles/{fn}", str(local))
                return "downloaded"
            except Exception as e:
                return f"failed: {e}"

        counts = {"downloaded": 0, "skipped": 0}
        with ThreadPoolExecutor(max_workers=download_workers(cfg)) as executor:
            futures = {executor.submit(_fetch, fn): fn for fn in filenames}
            for future in as_completed(futures):
                result = future.result()
                if result in counts:
                    counts[result] += 1
                else:
                    failed.append((futures[future], result))
        log.info(f"{zone_dir.name}: {len(filenames)} non-residential profiles, "
                 f"{counts['downloaded']} downloaded, {counts['skipped']} already present, "
                 f"{len(filenames) - sum(counts.values())} failed -> {profiles_dir}")

    if failed:
        examples = "\n".join(f"  {fn}: {r}" for fn, r in failed[:10])
        raise RuntimeError(f"{len(failed)} profile file(s) failed to download:\n{examples}\n"
                           f"Re-run data acquisition to retry; finished files are skipped.")


if __name__ == "__main__":
    cfg = load_config()
    download_metadata(cfg)
    ids = get_valid_building_ids(cfg)
    download_building_files(cfg, ids)
    if network_download_enabled(cfg):
        download_network(cfg)
    if network_profiles_enabled(cfg):
        download_network_profiles(cfg)
