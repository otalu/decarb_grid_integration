# Building-Grid Integration Pipeline

This Python-based pipeline is designed to integrate [DECARB](https://github.com/pduenas/decarb/tree/main), a Julia-based optimization model for planning building energy use, and various Julia-based electricity network optimization models, such as [OpenDSSDirect](https://github.com/dss-extensions/OpenDSSDirect.jl) and [GenX](https://genxproject.github.io/GenX.jl/stable/). The pipeline makes it easy to create bookend cases for DECARB to investigate the downstream network effects of various demand-side policies (such as tariffs or rebates) and relevant scenarios concerning residential energy use (such as heating electrification and proliferation of rooftop PV panels). It uses a backend of [ResStock building profiles](https://resstock.nlr.gov/) and [a combined synthetic dataset of the Texas transmission and distribution networks](https://electricgrids.engr.tamu.edu/combined-td-synthetic-dataset/). 

The residential buildings in an area of concern (such as a feeder or zone in a distribution model or an entire state in a transmission-generation model) are matched to the load buses in the synthetic grid based on their demand profiles, fitted with a simple thermal model based on their ResStock consumption profiles, and are optimized with DECARB to then fed back into OpenDSSDirect or GenX 
for network level analysis.

```
preprocess  ->  decarb-inputs  ->  DECARB (Julia)  ->  network-inputs  ->  OpenDSSDirect or GenX (Julia)  ->  results
```

The zone workflow (`scope: "zone"`, OpenDSS) is the maintained path. The statewide GenX path (`scope: "statewide"`) is not up to date.

---

## 1. Setup

**Python 3.10+**

```bash
pip install -r requirements.txt
```

**Julia + DECARB**: Julia 1.10+ and the DECARB repository, with its environment instantiated (`julia --project=<DECARB folder> -e "using Pkg; Pkg.instantiate()"`). Gurobi needs a license; HiGHS is free and installed with DECARB.

## 2. Input data

Downloaded automatically by the data acquisition stage. Everything is public, and no account or API key is needed:

| Data | Source | Saved to |
|---|---|---|
| ResStock metadata (`{STATE}_upgrade0.parquet`, plus a `.csv.gz` copy for manual inspection) | OEDI S3 `oedi-data-lake`, `resstock` settings in `config.yaml` | `in/` |
| ResStock 15-min building profiles | same | `in/residential_profiles/{STATE}/` |
| SMART-DS feeder network (`.dss` files for the zone) | OEDI S3, `smartds` settings in `config.yaml` | `paths.network_dir` (default `in/smartds/`) |
| SMART-DS profiles of the zone's non-residential (commercial) loads (`data_acquisition.download_network_profiles`) | same | `paths.network_dir/profiles/` |

Included in the repo:

| File | What it is | Location (setting) |
|---|---|---|
| DECARB input templates | Equipment catalogs and model settings for DECARB. Their costs and efficiencies are model inputs; see [docs/modeling_notes.md](docs/modeling_notes.md) section 4 | `in/decarb_templates/` (`template_dir` in `decarb.yaml`) |
| NEEP heat pump database | NEEP cold-climate ASHP product list export, used for heat pump selection. A replacement needs the `Brand Name`, `Status`, `HSPF (Region IV)` and the rated/min/max capacity and COP columns at 5/17/47/82/95 °F | `in/neep_database.csv` (`heatpump.neep_database`) |
| County shapefile | US county polygons (OpenDataSoft "US County Boundaries"), fields `name`, `statefp`, `countyfp`; used to assign feeder loads to counties | `in/us_county_boundaries/` (`paths.county_shapefile`) |

The building state and the network do not have to match. For example, NY or MA buildings on Texas feeders is a normal setup.

## 3. Configuration

| File | Controls |
|---|---|
| `config/config.yaml` | state/year/time zone, zone, paths, test mode, which preprocessing stages run, matching, heat pump sizing, date range and timestep |
| `config/decarb.yaml` | tariffs, electrification levels, rooftop solar, which DECARB input files to write |
| `config/network.yaml` | GenX settings (statewide only), interpolation, workers |

Settings a new user usually changes:

```yaml
state: "NY"                   # ResStock state of the buildings
timezone: "America/New_York"  # time zone of the buildings (TOU peak hours, DECARB output dates)
zone: {name: "P101U"}         # SMART-DS zone (feeder network) to use
#   zone.substations / zone.feeders: optional lists to run only part of the zone
paths:
  network_dir: "../in/smartds"  # on Windows keep the project path short (260-character path limit)
test_mode: true               # 600 buildings and one zone, for a quick end-to-end check
stages: {...: true}           # turn on the preprocessing stages you need
```

Settings that change results (heat pump sizing, matching limits, time zone, timestep, data locations) have no built-in defaults. If one is missing or invalid, the run stops with a message naming the key.

## 4. Running the pipeline

All commands are run from anywhere with the project's `src/run_pipeline.py`.

### 4.1 Preprocess

```bash
python src/run_pipeline.py preprocess
```

Runs the stages switched on under `stages:`, in this order:

| Stage | Reads | Writes |
|---|---|---|
| `data_acquisition` | S3 | metadata, building ID list (`in/resstock_bldg_ids_{state}_{year}.csv`, always the full list), building profiles, SMART-DS network (`data_acquisition.download_network`) |
| `thermal_model` | profiles, metadata | `out/thermal_model_outputs/{STATE}/baseline_internal_gains/regression_coeff_*.csv`, `skipped_buildings.csv` |
| `feeder_processing` | `network_dir/{zone}` `Loads.dss`/`Buscoords.dss`, county shapefile | `in/feeder_summaries.parquet` (residential loads with county and climate zone) |
| `building_matching` | feeder summaries, thermal results, metadata | `in/feeder_summaries_matched.parquet` (ResStock building for every load) |
| `schedules` | profiles, metadata | `in/schedules/{STATE}/`, `in/setpoint_timeseries/{STATE}/` |
| `heatpump_selection` | profiles, metadata, NEEP database | `out/heatpump_files/{STATE}/building_hp_summary.csv` |

Every stage stops with a clear error if its inputs are missing, instead of producing partial results. With `test_mode: true`, every stage uses the same `test_limit` lowest building IDs and only `zone.name`, and logs `TEST MODE: ...`.

`in/feeder_summaries*.parquet` hold one zone and state at a time. Switching `zone.name` or `state` means re-running feeder processing and matching.

To run only part of a zone, list substations and/or feeders under `zone.substations` / `zone.feeders` (folder names in `network_dir/{zone}/scenarios/base_timeseries/opendss/{substation}/{feeder}`; if both are set, their union). Feeder processing, matching (and so DECARB), the commercial profile download and the OpenDSS cases then cover only those feeders. Changing the selection means re-running feeder processing and matching; `decarb-inputs` and `network-inputs` stop if the matching file includes feeders outside the selection. `decarb-inputs` and `network-inputs` stop if the matched file is for a different zone.

### 4.2 DECARB inputs

```bash
python src/run_pipeline.py decarb-inputs
```

- Checks every required file first, and lists any building with missing inputs in `out/decarb_inputs_missing_files.csv`. Then it asks for a short study description and creates `out/studies/study_N/`.
- Writes `decarb/{tariff}_heat_{0|100}_solar_{0|100}/{bldg_id}/in/` for every matched building, plus copies of the three config files and the matching file.
- The thermal coefficients in `bdg_ii.csv` are written as fitted (15-min), with `pBtime = 0.25`; DECARB converts them to its own step length.
- `date_range` in `config.yaml` limits `tm.csv` to that period. Preprocessing always uses the full year.

### 4.3 Run DECARB

Outside this Python pipeline: edit the settings block at the top of `src/run_decarb_study.jl` (`STUDY_DIR`, `DECARB_DIR`, `N_WORKERS`, `SOLVER`, `GUROBI_LICENSE`) and run it with Julia. It loads the DECARB package from `DECARB_DIR` and solves with Gurobi (or the free HiGHS) at a 1% MIP gap and 300 s per building. DECARB writes `{bldg_id}/out/ts.csv` for each building; failures go to `failed_buildings.csv` in the case folder.

### 4.4 Network inputs

```bash
python src/run_pipeline.py network-inputs --study N
```

Builds `out/studies/study_N/opendss/{case}/`. It copies the zone's feeders from `network_dir`, and replaces each matched residential load with that building's DECARB profile (kW and reconstructed kvar). Non-residential (commercial) loads stay as in SMART-DS, with their SMART-DS profiles shifted to the DECARB clock (`smartds.utc_offset_hours`), resampled to the case timestep and written once per study to `opendss/smartds_profiles/`. The time axis follows the `timestep` and `date_range` saved in the study's own `config.yaml`. With `interpolation.enabled: true` in `network.yaml`, it also writes intermediate electrification/solar penetration levels (see below).

### 4.5 OpenDSS and results

```bash
python src/run_pipeline.py opendss --study N
python src/run_pipeline.py results --study N
```

The OpenDSS runner reads each case's step size and length from its `Master.dss`, so hourly and 15-min cases run the same way. Results go to `out/studies/study_N/opendss_results/{case}/zone_yearly.csv`.

### 4.6 Running everything on ORCD (MIT Engaging)

The `orcd/` folder submits every step as a SLURM job. It does not install anything; you need, in your home folder:

- this repository (e.g. `~/decarb_grid_integration`) and the DECARB repository (e.g. `~/decarb-main`, with `Pkg.instantiate()` done once);
- a Python 3.11+ environment with `requirements.txt` installed (e.g. `module load miniforge`, then `conda create -n decarb_grid python=3.12` and `pip install -r requirements.txt`);
- Julia, and Gurobi with your own license (or use `SOLVER="highs"`).

Fill in `orcd/env.sh` once: partition, mail address, DECARB folder, solver, and the lines that set up your Python environment, Julia and Gurobi in a job (`setup_python`, `setup_julia`, `setup_gurobi`). The other scripts find the repository from their own location and need no editing.

Edit `config/*.yaml` as usual, then submit from the login node:

```bash
bash orcd/submit_preprocess.sh 90
bash orcd/submit_decarb_inputs.sh 90 "study description"
bash orcd/submit_decarb.sh 90 [study_number]
bash orcd/submit_network.sh 90 [study_number]     # network-inputs, opendss, results
```

The first argument is the job's workers, 60 or 90 (60 → 1000G, 90 → 1500G, CPUs = workers + 1). In the Python jobs that worker count replaces the usual parallel workers of the downloads, thermal fit, `decarb-inputs` and OpenDSS (`PIPELINE_WORKERS`); run outside these scripts they keep their usual values.

or the whole chain at once, each job starting only if the previous one succeeded:

```bash
bash orcd/run_all.sh 90 "study description"
```

`run_all.sh` gives every job the same worker count. Without a study number, a job uses the latest study when it starts. Other worker counts are refused, and so are more than 100 DECARB workers with Gurobi (license limit). DECARB solves the cases in place in `out/studies/study_N/decarb/`, so `network-inputs` reads them directly. Job output goes to `out/logs/slurm/`, next to the pipeline logs in `out/logs/`. On `mit_preemptable` a job can be stopped early; submitting DECARB again skips buildings that already have `out/ts.csv`.

## 5. Time resolution

| Part | Resolution |
|---|---|
| ResStock data, thermal fit, schedules, heat pump sizing | always 15-min |
| DECARB inputs, OpenDSS cases, interpolation | `config.yaml` → `timestep` (`hourly` or `15min`) |
| GenX demand | hourly only |

## 6. Interpolation

DECARB runs only the 0% and 100% bookends of each dimension (electrification, solar). For an intermediate level, a seeded random set of buildings, weighted by how many loads each represents, takes its 100% profile and the rest keep their 0% profile. The sum is the zone demand at that level. Output: `out/studies/study_N/interpolation_summary.csv`.

## 7. Modeling notes

[docs/modeling_notes.md](docs/modeling_notes.md) explains where the values
handed to DECARB come from and why they differ from the raw data: ResStock
timestamp convention, thermal coefficient timestep, heat pump capacity
derating, and the DECARB templates.

## 8. Folder layout

```
config/                     config.yaml, decarb.yaml, network.yaml
src/                        pipeline stages; src/extras/ holds one-off analysis scripts
orcd/                       SLURM submit scripts for ORCD (env.sh = your settings)
in/                         downloaded and preprocessed inputs (not in git)
  smartds/                  SMART-DS feeder network, one folder per zone
out/
  logs/                     one log file per run_pipeline.py command ({command}_{date_time}.log)
  thermal_model_outputs/    thermal fits per state
  heatpump_files/           heat pump selection per state
  studies/
    study_log.csv           description of every study
    study_N/                one DECARB study: config snapshots, decarb/, opendss/, opendss_results/
```
