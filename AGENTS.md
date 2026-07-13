# AGENTS.md

Guidance for OpenCode sessions working in this repo. Every line below is something an agent would likely get wrong without it.

## What this is

Julia research code for energy coalition formation among buildings with battery storage. Model Predictive Control (MPC) over a horizon, solved with JuMP + SCS, coordinated via ADMM. No package manifest, no README, no CI, no test runner — pure scripts that `include` each other.

## Environment setup (non-obvious)

- **No `Project.toml` exists.** Scripts rely on a pre-populated global/default Julia environment. Verified: `julia -e 'using JuMP'` fails in this repo as-is.
- Required packages must be installed into the active env before anything runs:
  ```
  julia -e 'import Pkg; Pkg.add(["JuMP","SCS","Combinatorics","StatsBase","DataFrames","Dates","Statistics","LinearAlgebra","Plots","Measures","Printf","Random"])'
  ```
- **Threading:** `privacy_focussed_coals` (MPC variant) uses `@threads` in `Coalition.jl:312`. Run with `julia --threads auto` (or set `JULIA_NUM_THREADS`); otherwise it runs serially with no warning — a silent perf cliff.
- `@eval CSV.Parsers import Base.Ryu: writeshortest` (in `load_EMS_data.jl:103` and all `experiments/*.jl`) is a fragile CSV.jl internal workaround. If it errors on a CSV.jl upgrade, that line is the suspect.

## Data (required, gitignored)

`data/`, `cleaned_data/`, `results/` are gitignored. Loaders read:
- `cleaned_data/{1..70}.csv` — per-building horizons (committed copy is ~17MB each).
- `data/metadata.csv`, `data/edf_prices.csv` — battery params + 96-step buy/sell prices.
- Raw `data/{i}.csv` are 169–333MB each and only needed to regenerate `cleaned_data/` via `clean_data()` in `load_EMS_data.jl:102`.

Without `cleaned_data/` present, every script and test fails at load. A fresh clone has none of it.

## Entry points and include order

Top-level scripts, run directly: `julia --threads auto run_MPC.jl`.
- `run_MPC.jl` — MPC / receding-horizon path (`MPC_Building`).
- `run_open.jl` — single-shot path (`Building`).

Both `include` in this exact order, which matters (`Coalition.jl` depends on types from `Buildings.jl`; `MPC_optimiser.jl` re-includes `Buildings.jl`):
```
Buildings.jl → MPC_optimiser.jl → Coalition.jl → load_EMS_data.jl → plotting.jl
```

`experiments/*.jl` use `include("../...")` and must be run from their own directory (or the repo root with the `../` paths resolving).

## Critical: `opt` is an implicit global

`ADMM_build_opt_init` (`MPC_optimiser.jl:222`) and `ADMM_build_opt_solve` (`MPC_optimiser.jl:268`) reference `opt` via `energy_cost_k(opt,...)` but `opt` is **not a parameter**. They depend on a global `opt = MPC_optimiser(energy_cost', energy_sale')` defined by the caller. `test_regression.jl:88` explicitly uses `global opt = ...`.

Rule: any new entry point or refactor that touches ADMM must define `opt` in the same scope before calling. Breaking this fails at solve time with an `UndefVarError`, not at load.

## Two parallel type hierarchies — don't mix them

- `Building` (`Buildings.jl:3`): single-shot, used by `run_open.jl` and `load_from_CSV`.
- `MPC_Building` (`Buildings.jl:24`): horizon-based, used by `run_MPC.jl`, `experiments/`, `MPC_load_from_CSV`.

`find_opt_coal`, `bottom_up_full_info`, `privacy_focussed_coals` each have **two overloads** in `Coalition.jl` — a 2-arg `Building` version and a 5-arg `MPC_Building` version (`..., k, num_look_ahead, receding_horizon=false`). Passing the wrong building type silently dispatches to the wrong method. Check the call site's `num_builds`/`num_look_ahead` arguments to know which you're in.

## `MPC_Building.SoC` is mutated in place

`optimise` and `coal_MPC` write `b.SoC[k+1] = ...` directly into the building object (`MPC_optimiser.jl:417`, `:519`, `:535`). Reusing the same `buildings` vector across runs **without resetting `SoC` to zeros** gives stale-state results. When re-running, rebuild via `MPC_load_from_CSV` or zero out `SoC` first.

NaN guards exist in `coal_MPC` (`isnan(next_soc) ? b.SoC[k] : ...`) but not in `optimise` — a non-converged SCS solve there will propagate NaN.

## Tests are bespoke scripts (no `Pkg.test`)

`test_regression.jl` — regression for `privacy_focussed_coals`:
- First record baselines: `julia test_regression.jl --baseline` (writes `results/baseline_regression.txt`, `results/baseline_regression_k40.txt`).
- Then verify: `julia test_regression.jl`.
- Run one case: `julia test_regression.jl --case 1`.
- Case 1 (singleton path) is deterministic — asserts exact coalition match, tight cost tol (1e-3 rel).
- Case 2 (merges, k=40) is RNG-dependent — loose cost tol (20% rel), no exact coalition match. **Per-seed results vary** because the sampler consumes a variable number of RNG draws.
- Exits 0 on PASS, 1 on FAIL.

`test_sampling.jl` — validates the Phase A sampling refactor. Includes only `sampling.jl`. Checks validity + distributional equivalence (total-variation distance ≤ 0.10) between `sample_coals_old` and `sample_coals_new` over 2000 seeds. Standalone, no data files needed.

## `find_opt_coal` is exhaustive — do not scale it

`find_opt_coal` enumerates **all partitions** of the building set. `run_open.jl:108` comments "takes about 20 mins on laptop to execute 8" and gates it behind `if num_builds <= 8`. Removing that guard for larger `num_builds` will hang. Use `bottom_up_full_info` or `privacy_focussed_coals` for larger sets.

## Experiments write incrementally and swallow errors

`experiments/*.jl` loop over `num_builds` or `max_coal_size`, appending rows to `results/*.csv` **inside the loop** (so partial results survive a crash) and wrap each iteration in `try/catch` that prints `"Failed at N"` and continues. Implications:
- `results/*.csv` rows may be partial / out of order — don't assume a complete sweep.
- Failures are silent except for a stdout line; check for "Failed at" in output.
- They 3-destructure `MPC_load_from_CSV` which returns 4 values — Julia drops the 4th silently. Fine, but don't expect `timestamps` there.

## Plotting

`plotting.jl` uses `Plots.jl` with `fontfamily="Computer Modern"` and saves PDFs to `results/`. LaTeX fonts must be installed or it silently falls back to defaults. Plot functions are called from entry scripts; not run standalone.

## `backup/`

`backup/Coalition.jl.orig` and `backup/MPC_optimiser.jl.orig` are pre-refactor snapshots, committed for reference. Nothing loads them. Useful if you need to see what a function looked like before the ADMM caching / Phase A sampling changes.

## Committed vs gitignored results

`results/baseline_regression*.txt` are **committed** (needed by `test_regression.jl`). Everything else in `results/` (CSVs, PDFs) is gitignored and regenerated by experiments.

## Quick verification after edits

```
julia --threads auto test_regression.jl --baseline   # if baselines need refresh
julia --threads auto test_regression.jl              # must end "Overall: PASS"
julia test_sampling.jl                                # must end "Overall: PASS"
```
If you only touched `sampling.jl`, `test_sampling.jl` is the focused check. If you touched `Coalition.jl` or `MPC_optimiser.jl`, run `test_regression.jl` too.
