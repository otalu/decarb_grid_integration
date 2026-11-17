# Modeling notes

Why the numbers the pipeline hands to DECARB differ from the raw data, and
where each value comes from. Each section gives the rule, the reason, and the
code that implements it.

## 1. ResStock timestamps

ResStock 2025 release 1 individual-building timeseries
(`resstock_amy2018_release_1`) are 15-minute data whose timestamps mark the
**end** of each interval, in **Eastern Standard Time (UTC-5) for every state**,
with no daylight saving. The first row is `00:15`, the last is `00:00` on
January 1 of the next year.

The release README does not state this; it was verified on the data: the
direct-normal-irradiance-weighted time of day is about 12:00 for New York and
about 13:45 for a San Antonio building (longitude -98.46, solar noon about
12:35 Central = 13:35 Eastern), the same in January and July.

- Legacy DECARB read `tm.csv` timestamps as local clock time in `pBtz`, with
  daylight saving. The pipeline labeled the data accordingly and shifted it by
  one hour from the spring DST date. For buildings outside Eastern time this
  left `tm.csv` 1 hour (winter) to 2 hours (summer) off from the data. The
  shift was never undone in November, so from Nov 4 on the TOU peak hours sat
  1 hour late. `pBtz` was never set, so every building got the template's
  `America/Chicago`.
- The current DECARB reads `tm.csv` timestamps as **UTC period ends**
  (`docs/inputs_outputs.md` in DECARB), so `tm.csv` gets UTC = ResStock
  timestamp + 5 hours for every state, and `cfg.csv` `pP0` = first timestamp
  minus one step.

Rules tied to the wall clock use local time (`config.yaml` -> `timezone`, with
daylight saving) at the start of each step: TOU peak windows and the ResStock
heating/cooling unavailable periods. `pBtz` is set to the same time zone; in
the current DECARB it only sets the time zone of the `Date` column in its
outputs.

**SMART-DS profiles (non-residential loads).** The OpenDSS cases replace the
residential loads with DECARB results and keep every other load (commercial)
with its SMART-DS profile. Those profiles are per-unit, 15-minute, 35040 rows,
with no timestamps. SMART-DS does not document their clock; for Full_Texas it
was inferred from the data: the weekday morning ramp is 1 h earlier in summer
than in winter (standard time, no DST), and the evening lighting jump follows
Dallas sunset in Central standard time (+0.1 to +0.2 h in the clean months). So
row 0 is Jan 1 00:00-00:15 CST (UTC-6), one hour after the ResStock/DECARB
row 0 (00:00 EST, UTC-5). `network-inputs` rolls them one hour
(`smartds.utc_offset_hours` = -6) so both refer to the same UTC instants, then
resamples to the case timestep and cuts to `date_range`. Before this, the
OpenDSS cases contained the residential loads only.

## 2. Thermal model coefficients (`bdg_ii.csv`)

The thermal model (`src/fit_thermal_model.py`) is fitted on the 15-minute
ResStock data:

```
T_in(t) = T_in(t-1) + k1*[T_out(t-1) - T_in(t-1)] + k2*Q_rad(t) + k3*Q_net(t-1)
```

so `k1, k2, k3` describe one 15-minute step. A model with longer steps needs
larger coefficients. Applying the 15-minute model n times with constant inputs
gives, exactly:

```
k1_n = 1 - (1 - k1)^n
k2_n = k2 * k1_n / k1        k3_n = k3 * k1_n / k1      (factor n when k1 = 0)
```

n = 4 for hourly steps (about 4x the 15-minute values).

- Legacy DECARB applies `pBk1-3` once per model step as given, so the pipeline
  converted them for the configured `timestep` (`convert_thermal_coeffs`,
  removed with the move to the current DECARB). Hourly studies made before
  that conversion was added used 15-minute coefficients at hourly steps (the
  building responded about 4x too slowly).
- The current DECARB takes the calibration interval as an input (`pBtime`, in
  hours) and does this conversion itself with the same formula
  (`thermal_response_coefficients` in `thermal_load.jl`), so `bdg_ii.csv` gets
  the 15-minute fit unchanged with `pBtime = 0.25`.

`pBk2` and `pBk3` are written x1000 (W to kW).

**Internal gains.** `k3` multiplies heating/cooling plus internal gains, so the
fit uses the same internal gains DECARB computes
(`pBihgp * occupants + pBihgl * pQlight + pBihge * pQequip`), built by
`src/internal_gains.py` for both the fit and `tm.csv`: all electric lighting,
electric equipment (electricity total minus electric heating, cooling, hot
water, EV and lighting), and 0.1 kW per occupant (factors from the `bdg_i.csv`
template). Gas, propane and oil end uses other than heating and hot water are
not modeled (DECARB bills `pQlight`/`pQequip` as electricity). Until
`THERMAL_MODEL_VERSION` 2 the fit used interior lighting, plug loads and the
ResStock people heat rate while DECARB got the larger loads (0.66 vs 0.44 kW
on average for 600 NY buildings), which left the simulated indoor temperature
about 0.6 C too warm on average; with matching gains the median bias is 0.0 C
and the median free-running error 2.2 C (2.3 C before).

## 3. Heat pump capacity derating (Mitsubishi rows in `hvac.csv`)

`src/heatpump_selection.py` fits each NEEP unit's capacity against outdoor
temperature relative to the **design temperature** `T_d = 8.33 C` (47 F):

```
heating: C * (1 - k * (T_d - T_out))        cooling: C * (1 - k * (T_out - T_d))
```

`C` is the rated capacity at `T_d`, `k` the loss per degree. These are what the
heat pump summary stores and what heat pump sizing uses.

DECARB derates capacity from the **indoor** temperature instead
(`hvac_units.jl`: output share <= 1 - pHVmx_ * max(T_in - T_out, 0)), and
treats `pHVmx`/`pACmx` as the capacity where that derating starts (outdoor =
indoor). Written as-is, the fit would start losing capacity about 12 C too
early: for the NY heat pumps, about 45% of rated capacity at -10 C instead of
the fitted 67%.

`create_decarb_inputs.py` therefore re-expresses the same curve in DECARB's
form (`capacity_for_indoor_reference`), for a reference indoor temperature
`T_in`:

```
heating: pHVmx = C * (1 + k*(T_in - T_d))      pHVmx_ = k / (1 + k*(T_in - T_d))
cooling: pACmx = C * (1 - k*(T_in - T_d))      pACmx_ = k / (1 - k*(T_in - T_d))
```

This reproduces the fitted curve exactly when DECARB's indoor temperature
equals `T_in`; within the comfort band it is off by about 1.5% of capacity per
degree. `T_in` is each building's average comfort-band midpoint
((min + max comfort temperature) / 2) over its heating-mode periods for
heating and cooling-mode periods for cooling, from its setpoint file
(`comfort_reference_temps`).

The written `pHVmx` is therefore the fitted capacity read at the indoor
temperature, not at 8.33 C (about x1.23 for heating and x0.89 for cooling in
NY). COPs, COP losses and the design temperature are written unchanged; the
efficiency loss in DECARB is measured from the design temperature, as fitted.

Only the Mitsubishi row is converted. The generic HVAC rows keep the template's
capacity losses (see 4).

**Known limitation:** the current DECARB caps `pHVmx_` and `pACmx_` at 0.02 per
degree (`load_hvac.jl`, "to avoid infeasibility"). About 5% of the NY heat
pumps have a converted heating loss above 0.02 and are capped, which slightly
overstates their cold-weather capacity.

## 4. DECARB input templates

Every DECARB input file starts from a template; `create_decarb_inputs.py`
overwrites only the building-specific fields (location, areas, capacities, the
installed equipment in `sp.csv`, the Mitsubishi row, `pTin0`). Everything else
(equipment efficiencies, costs, penalty costs, investment settings) comes from
the template unchanged, so the templates are model inputs, not just formats.

`in/decarb_templates/` holds the templates in the current DECARB's format,
converted from the legacy templates
(`D:/shared/building_grid_integration/in/decarb_input_templates/` on SAMWISE)
with all values kept except:

| File | Change |
|---|---|
| `abs.csv` | renamed `abp.csv`; column `hr` -> `fcf` (values 0) |
| `chp.csv`, `wh.csv` | column `hr` -> `fcf` = 1/`hr`, i.e. the efficiency in the equipment name (76% AFUE -> 0.76). Legacy DECARB computed fuel as output / `hr`, so a 76% furnace burned less fuel than the heat it delivered; the current DECARB computes output / `fcf`. Fuel use and cost of fuel-fired equipment are therefore higher than in legacy runs by a factor `hr^2`. Electric water heaters: `fcf = 0` (not used for electric units). |
| `topo.csv` | columns `fuel, mx, hr` removed |
| `sp.csv` | columns `pABS*` -> `pABP*`; the `Electricity ASHP SEER 20/22/24` entries in `pHVAC0` removed (set to 0), see below |
| `bdg_i.csv` | removed `pBventy, pBvent, pBslab, pBtslab, pBkslab, pBpslab, pBcpslab`; `pBabs` -> `pBabp`; added `pBalb = 0.3` (PV albedo, new input) |
| `in.csv` | renamed `cfg.csv`; `pGco2`/`pLco2` 53.07/73.16 kg/MMBtu -> 0.1811/0.2496 kg/kWh (DECARB's unit; emissions are reported only, not in the objective) |

**HVAC catalog (`hvac.csv`).** The COPs (`pHVeff`, `pACeff`) and COP losses
(`pHVeff_`, `pACeff_`) of the generic ResStock equipment types are the lumped
per-type fits from the ERCOT buildings
(`Z:/ercot_project/src/resstock_cop_capacity_estimation.py`, outputs in
`ercot_project/out/cop_capacity_analysis/cop_fit_by_hvac_category.csv`):
COP = intercept at zero temperature difference, loss = fitted slope, rounded to
2-3 decimals. The capacity losses (`pHVmx_ = 0.02`, `pACmx_ = 0.03`) are
placeholders: the per-type capacity fits in `capacity_fit_by_hvac_category.csv`
are in kW per degree and were not carried into the catalog. Divided by the
fitted capacity, several give 0.12-0.26 per degree (e.g. electric furnace
0.18), too steep to be capacity derating, so they were not used.

Electric resistance heating (`Electric Baseboard`, `Electric Boiler`,
`Electric Furnace`, `Electric Wall Furnace`) has all four losses set to 0
(`pHVmx_`, `pACmx_`, `pHVeff_`, `pACeff_`): resistance heat does not lose
capacity or efficiency with outdoor temperature. Changed from the legacy
template: `pHVmx_` 0.02 -> 0 and `pACmx_` 0.03 -> 0 on all four rows (`pACmx_`
has no effect, they have no cooling), and `Electric Boiler` `pHVeff_`
0.005 -> 0.

**SEER 20/22/24 heat pumps (left out for now).** The legacy `sp.csv` template
listed `Electricity ASHP SEER 20/22/24` with `pHVACyn = YES`, but the current
`hvac.csv` catalog has no rows for them, and the current DECARB rejects
selections missing from the catalog. They were never actually offered:
`create_sp_file` sets `pHVACyn = NO` for every row. The only values for them
(`hvac_old.csv`) are rating / 3.412 COPs with flat 0.05 losses, not on the same
basis as the fitted rows, so they are left out until fitted-basis values exist.

**Location:** `in/decarb_templates/` (`decarb.yaml` -> `template_dir`), tracked
in git so a new user gets them with the code.
