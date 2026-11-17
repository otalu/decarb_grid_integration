"""
Build workshop-style variations of DECARB's own `examples/annual` case
(Houston, hourly, full year 2018), like build_workshop_cases.py does for a
pipeline building. The example has no pipeline study behind it, so every case is
made by editing the example's in/ files directly:

  base_‹tariff›            the example's HVAC (ASHP SEER 13) and electric water heater;
                           its PV panels and battery removed
  heat_pump_‹tariff›       Mitsubishi ASHP (from the example's hvac.csv) replaces the ASHP
  hp_backup_‹tariff›       heat pump + gas furnace 80% AFUE (already linked in topo.csv) at
                           BACKUP_FRACTION of the heat pump heating capacity, fuel prices
                           (pGcost, pLcost) = FUEL_PRICE
  hp_solar_bess_‹tariff›   heat pump + the example's PV panels and battery

‹tariff› is each tariff in decarb.yaml (flat, flat_cap, tou, tou_cap): pQcostBuy,
pQcostSell, pQmx and pQmxCost in tm.csv are rewritten the way create_decarb_inputs
does it (TOU peak windows on local time from bdg_i.csv pBtz; tm.csv pP are UTC
period ends). The example is hourly only, so there are no 15-min cases.
Writes a source.txt per case and a bdg_path.txt listing every case.

Usage:
    python src/extras/build_annual_example_cases.py --decarb-dir C:/.../decarb
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from config.loader import load_decarb_config  # noqa: E402

TARIFFS = ["flat", "flat_cap", "tou", "tou_cap"]
HEAT_PUMP_PREFIX = "Mitsubishi ASHP"
BACKUP_HEATER = "Natural Gas Fuel Furnace 80% AFUE"
BACKUP_FRACTION = 0.2
FUEL_PRICE = 0.23                          # $/kWh, backup cases


def read_str(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_kv(path):
    return pd.read_csv(path, header=None, index_col=0, dtype=str, keep_default_na=False)


def apply_tariff(tm_path, tariff, tz):
    """Same rules as create_decarb_inputs.generate_tm_data."""
    tm = pd.read_csv(tm_path)
    ends = pd.to_datetime(tm.pP)
    step = ends.diff().median()
    local = (ends - step).dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
    if tariff["type"] == "flat":
        tm["pQcostBuy"] = tariff["rate"]
    else:
        times = local.dt.time
        peak = pd.Series(False, index=tm.index)
        for win in tariff.get("peak_windows", []):
            ws = datetime.strptime(win["start"], "%H:%M").time()
            we = datetime.strptime(win["end"], "%H:%M").time()
            peak |= ((times >= ws) & (times < we)) if ws < we else ((times >= ws) | (times < we))
        tm["pQcostBuy"] = tariff["rate"]
        tm.loc[peak, "pQcostBuy"] = tariff["peak_price"]
    tm["pQcostSell"] = tariff.get("sell_rate", 0.18)
    cap = tariff.get("capacity_charge", {"enabled": False})
    tm["pQmx"] = 1 if cap.get("enabled", False) else 0
    tm["pQmxCost"] = cap["charge_value"] if cap.get("enabled", False) else 0.0
    tm.to_csv(tm_path, index=False)


class Builder:
    def __init__(self, src_in, out_dir, tariffs):
        self.src_in, self.out_dir, self.tariffs = src_in, out_dir, tariffs
        self.tz = read_kv(src_in / "bdg_i.csv").loc["pBtz", 1]
        sp = read_str(src_in / "sp.csv")
        self.pv = sp.loc[sp.pPVz0 != "0", ["pPV0", "pPVz0"]]
        self.bess = sp.loc[sp.pBESSz0 != "0", ["pBESS0", "pBESSz0"]]
        hvac = read_str(src_in / "hvac.csv")
        hp = hvac[hvac.ty.str.startswith(HEAT_PUMP_PREFIX)]
        if len(hp) != 1:
            raise ValueError(f"expected one {HEAT_PUMP_PREFIX} row in hvac.csv, found {len(hp)}")
        self.hp_name, self.hp_heat_kw = hp.ty.iloc[0], float(hp.pHVmx.iloc[0])
        self.existing_hvac = sp.loc[sp.pHVACz0 == "1", "pHVAC0"].tolist()
        self.cases = []

    def copy(self, name, tariff):
        d = self.out_dir / name / "in"
        if d.parent.exists():
            shutil.rmtree(d.parent)
        shutil.copytree(self.src_in, d)
        apply_tariff(d / "tm.csv", self.tariffs[tariff], self.tz)
        self.cases.append(name)
        return d

    def edit_sp(self, d, pv_bess, heat_pump):
        sp = read_str(d / "sp.csv")
        if not pv_bess:
            sp.loc[sp.pPVz0 != "0", "pPVz0"] = "0"
            sp.loc[sp.pBESSz0 != "0", "pBESSz0"] = "0"
        if heat_pump:
            sp.loc[sp.pHVACz0 == "1", "pHVACz0"] = "0"
            # as create_decarb_inputs: the heat pump takes the first row
            sp.loc[0, ["pHVAC0", "pHVACz0", "pHVACyn"]] = [self.hp_name, "1", "NO"]
        sp.to_csv(d / "sp.csv", index=False)
        return sp

    def add_backup(self, d):
        kw = BACKUP_FRACTION * self.hp_heat_kw
        sp = read_str(d / "sp.csv")
        if (sp.pCHP0 == BACKUP_HEATER).sum() != 1:
            raise ValueError(f"{BACKUP_HEATER} not in sp.csv")
        sp.loc[sp.pCHP0 == BACKUP_HEATER, ["pCHPz0", "pCHPyn"]] = ["1", "NO"]
        sp.to_csv(d / "sp.csv", index=False)
        chp = read_str(d / "chp.csv")
        if (chp.ty == BACKUP_HEATER).sum() != 1:
            raise ValueError(f"{BACKUP_HEATER} not in chp.csv")
        chp.loc[chp.ty == BACKUP_HEATER, "mx"] = repr(kw)
        chp.to_csv(d / "chp.csv", index=False)
        if not (read_str(d / "topo.csv").up == BACKUP_HEATER).any():
            raise ValueError(f"{BACKUP_HEATER} is not linked to the building in topo.csv")
        tm = pd.read_csv(d / "tm.csv")
        tm["pGcost"] = FUEL_PRICE
        tm["pLcost"] = FUEL_PRICE
        tm.to_csv(d / "tm.csv", index=False)
        return kw

    def note(self, d, text):
        (d.parent / "source.txt").write_text(
            text.strip() + f"\nFrom DECARB examples/annual/in; tariff columns of tm.csv rewritten "
                           f"from decarb.yaml (TOU on local time, {self.tz}).\n")

    def build(self):
        existing = " + ".join(self.existing_hvac)
        pv = ", ".join(f"{r.pPVz0} x {r.pPV0}" for r in self.pv.itertuples())
        bess = ", ".join(f"{r.pBESSz0} x {r.pBESS0}" for r in self.bess.itertuples())
        for t in TARIFFS:
            d = self.copy(f"base_{t}", t)
            self.edit_sp(d, pv_bess=False, heat_pump=False)
            self.note(d, f"DECARB annual example: existing {existing}, PV and battery removed. "
                         f"Tariff: {t}. Hourly, full year 2018.")
        for t in TARIFFS:
            d = self.copy(f"heat_pump_{t}", t)
            self.edit_sp(d, pv_bess=False, heat_pump=True)
            self.note(d, f"DECARB annual example: {self.hp_name} replaces {existing}; PV and battery "
                         f"removed. Tariff: {t}. Hourly, full year 2018.")
        for t in TARIFFS:
            d = self.copy(f"hp_backup_{t}", t)
            self.edit_sp(d, pv_bess=False, heat_pump=True)
            kw = self.add_backup(d)
            self.note(d, f"DECARB annual example: {self.hp_name} plus a {BACKUP_HEATER} as backup at "
                         f"{BACKUP_FRACTION:.0%} of the heat pump heating capacity ({kw:.2f} kW); "
                         f"fuel prices (pGcost, pLcost) {FUEL_PRICE} $/kWh. PV and battery removed. "
                         f"Tariff: {t}. Hourly, full year 2018.")
        for t in TARIFFS:
            d = self.copy(f"hp_solar_bess_{t}", t)
            self.edit_sp(d, pv_bess=True, heat_pump=True)
            self.note(d, f"DECARB annual example: {self.hp_name} replaces {existing}; the example's "
                         f"PV ({pv}) and battery ({bess}) kept. Tariff: {t}. Hourly, full year 2018.")
        rel = Path("examples") / self.out_dir.name
        (self.out_dir / "bdg_path.txt").write_text("".join(f"{(rel / n).as_posix()}\n" for n in self.cases))
        return self.cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decarb-dir", type=Path, required=True, help="DECARB repository folder")
    ap.add_argument("--out-name", default="annual_updates", help="output folder under examples/")
    ap.add_argument("--decarb-yaml", type=Path, default=PROJECT_ROOT / "config" / "decarb.yaml",
                    help="pipeline decarb.yaml with the tariff definitions")
    args = ap.parse_args()
    tariffs = {t["name"]: t for t in load_decarb_config(args.decarb_yaml)["tariffs"]}
    missing = [t for t in TARIFFS if t not in tariffs]
    if missing:
        raise SystemExit(f"tariffs missing from {args.decarb_yaml}: {missing}")
    out = args.decarb_dir / "examples" / args.out_name
    cases = Builder(args.decarb_dir / "examples" / "annual" / "in", out, tariffs).build()
    print(f"{len(cases)} cases written to {out}")


if __name__ == "__main__":
    main()
