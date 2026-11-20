import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

_config = None


def job_workers(default):
    """Parallel workers for a step: PIPELINE_WORKERS if set (the orcd/ submit
    scripts set it to the job's worker count), else the step's usual value."""
    n = os.environ.get("PIPELINE_WORKERS")
    if n is None:
        return default
    if not n.isdigit() or int(n) < 1:
        raise ValueError(f"PIPELINE_WORKERS must be an integer >= 1, got {n!r}")
    return int(n)


def load_config(config_path=None):
    global _config
    if _config is not None and config_path is None:
        return _config

    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    config_dir = config_path.parent

    env_file = config_dir / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass

    with open(config_path, encoding="utf-8-sig") as f:   # tolerate a BOM from Windows editors
        cfg = yaml.safe_load(f)

    for key in ("project_root", "input_dir", "output_dir", "network_dir", "county_shapefile"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            p = config_dir / p
        cfg["paths"][key] = p

    hp_cfg = cfg.get("heatpump") or {}
    if hp_cfg.get("neep_database"):
        p = Path(hp_cfg["neep_database"])
        hp_cfg["neep_database"] = p if p.is_absolute() else config_dir / p

    _config = cfg
    return cfg


_decarb_config = None


def load_decarb_config(config_path=None):
    global _decarb_config
    if _decarb_config is not None and config_path is None:
        return _decarb_config

    if config_path is None:
        config_path = Path(__file__).parent / "decarb.yaml"
    else:
        config_path = Path(config_path)

    config_dir = config_path.parent

    with open(config_path, encoding="utf-8-sig") as f:   # tolerate a BOM from Windows editors
        dcfg = yaml.safe_load(f)

    template_dir = Path(dcfg.get("template_dir", "../in/decarb_templates"))
    if not template_dir.is_absolute():
        template_dir = config_dir / template_dir
    dcfg["template_dir"] = template_dir

    # outputs.in (legacy in.csv) was renamed outputs.cfg (cfg.csv)
    outputs = dcfg.get("outputs") or {}
    if "in" in outputs:
        if "cfg" in outputs:
            raise ValueError("decarb.yaml outputs has both 'in' and 'cfg'; remove 'in'")
        import logging
        logging.getLogger(__name__).warning(
            "decarb.yaml outputs 'in' is deprecated; rename it to 'cfg' (cfg.csv)")
        outputs["cfg"] = outputs.pop("in")

    _decarb_config = dcfg
    return dcfg


_network_config = None


def load_network_config(config_path=None):
    global _network_config
    if _network_config is not None and config_path is None:
        return _network_config

    if config_path is None:
        config_path = Path(__file__).parent / "network.yaml"
    else:
        config_path = Path(config_path)

    config_dir = config_path.parent

    with open(config_path, encoding="utf-8-sig") as f:   # tolerate a BOM from Windows editors
        ncfg = yaml.safe_load(f)

    # Resolve relative paths in genx section
    genx = ncfg.get("genx", {})
    for key in ("base_case", "bus_mapping"):
        val = genx.get(key)
        if val is not None:
            p = Path(val)
            if not p.is_absolute():
                genx[key] = config_dir / p
            else:
                genx[key] = p

    _network_config = ncfg
    return ncfg


def get_path(cfg, *parts):
    return cfg["paths"]["project_root"].joinpath(*parts)


def resolve_hour_indices(cfg):
    """Return a sorted array of 0-based hour indices (into 8760) for the
    configured date_range.  Handles year-wrap when start > end.
    Returns None when the range covers the full year."""
    dr = cfg.get("date_range")
    if dr is None:
        return None
    year = cfg["year"]
    start_str = dr.get("start", "Jan 1")
    end_str = dr.get("end", "Dec 31")

    start_dt = datetime.strptime(f"{start_str} {year}", "%b %d %Y")
    end_dt = datetime.strptime(f"{end_str} {year}", "%b %d %Y")
    end_dt = end_dt.replace(hour=23)

    jan1 = datetime(year, 1, 1)
    start_hour = int((start_dt - jan1).total_seconds() // 3600)
    end_hour = int((end_dt - jan1).total_seconds() // 3600)

    if start_hour <= end_hour:
        indices = np.arange(start_hour, end_hour + 1)
    else:
        indices = np.concatenate([
            np.arange(start_hour, 8760),
            np.arange(0, end_hour + 1),
        ])

    if len(indices) == 8760:
        return None
    return indices


TIMESTEPS = ("hourly", "15min")


def resolve_timestep(cfg, dcfg=None, source="config.yaml"):
    """DECARB-onward time resolution: config.yaml 'timestep'. An older
    decarb.yaml 'timestep' is still accepted (with a warning) when config.yaml
    has none; if both are set they must agree."""
    import logging
    log = logging.getLogger(__name__)
    timestep = cfg.get("timestep")
    old = (dcfg or {}).get("timestep")
    if timestep is None and old is not None:
        log.warning("'timestep' in decarb.yaml is deprecated; move it to config.yaml "
                    "(below date_range)")
        timestep, source = old, "decarb.yaml"
    elif timestep is not None and old is not None and old != timestep:
        raise ValueError(f"timestep is {timestep!r} in config.yaml but {old!r} in "
                         f"decarb.yaml; remove it from decarb.yaml")
    if timestep not in TIMESTEPS:
        raise ValueError(f"timestep in {source} must be one of {', '.join(TIMESTEPS)}, "
                         f"got {timestep!r}")
    return timestep


def resolve_step_indices(cfg, steps_per_hour):
    """Like resolve_hour_indices, but indices into a year of steps_per_hour
    steps per hour (4 for 15-min data). Returns None for the full year."""
    hours = resolve_hour_indices(cfg)
    if hours is None:
        return None
    return (hours[:, None] * steps_per_hour + np.arange(steps_per_hour)).ravel()
