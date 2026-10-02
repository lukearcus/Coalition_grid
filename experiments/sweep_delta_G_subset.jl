# =============================================================================
# delta_G sweep on a high-complementarity subset of the synthetic dataset.
#
# The synthetic data has 30 buildings across 10 archetypes (solar, wind,
# office, industrial, residential, etc.) with 4-tier TOU prices (buy-sell
# spread 0.05–0.30) and 7 days × 96 steps = 672 rows. Mean pairwise
# complementarity is ~39% (vs 11% for the original data).
#
# Subset: 6 buildings chosen for maximum temporal complementarity:
#   1  = solar farm     (midday seller)
#   10 = industrial     (always buyer, flat)
#   16 = data center    (always buyer, high)
#   25 = coastal wind   (overnight seller, 3am peak)
#   28 = hilltop wind   (evening seller, 5pm peak)
#   7  = residential    (AM/PM buyer, double peak)
#
# This gives 3 net sellers with staggered generation profiles + 3 net buyers
# with distinct consumption patterns — the conditions where coalition
# formation is most valuable and the privacy/cost tradeoff most visible.
#
# ============ Configuration (edit here or set via ENV vars) ============
#   NUM_REPEATS    (default 10)  repeats per (window, delta_G)
#   DELTA_G_POINTS (default 10)  log-spaced points from 0.1 to 10000
#                                (plus delta_G=0 deterministic anchor)
#
# ============ Windows ============
# Two windows: day 0 (weekday — schools/offices active) and day 5 (weekend —
# different demand pattern). Each window recomputes its own decentralised
# baseline.
#
# ============ Known issue (worked around) ============
# coal_MPC indexes buy/sell by b.id assuming id==position
# (MPC_optimiser.jl:525,527,540,542). The subset IDs [1,10,16,...] would
# crash, so slice_window re-IDs buildings to 1..N. Fix the indexing bug
# separately and this workaround can be dropped.
#
# ============ Run ============
#   julia --threads auto experiments/sweep_delta_G_subset.jl
# (from repo root -- loaders read synthetic_data/ relative to cwd)
# =============================================================================

try
    using CSV
catch
    using CSV
end
@eval CSV.Parsers import Base.Ryu: writeshortest
using DataFrames
using Random
using Statistics
include("../Buildings.jl")
include("../MPC_optimiser.jl")
include("../Coalition.jl")
include("../load_EMS_data.jl")

# ---------- Configuration ----------
# High-complementarity subset: 3 net sellers with staggered generation
# (solar midday, coastal wind overnight, hilltop wind evening) + 3 net buyers
# (industrial 24/7, data center 24/7, residential AM/PM peaks).
const SUBSET_IDS = [1, 10, 16, 25, 28, 7]
const num_builds = length(SUBSET_IDS)
const max_coal_size = 6
const num_ahead = 8
const receding_horizon = false
const N_STEPS = 96

# ENV overrides with sensible defaults.
const num_repeats    = parse(Int, get(ENV, "NUM_REPEATS", "10"))
const delta_g_points = parse(Int, get(ENV, "DELTA_G_POINTS", "10"))

# delta_G grid: 0.0 (deterministic argmax anchor) + log-spaced 0.01..1000.
# 0.01..1000 spans greedy (small) -> softmax (~1-10) -> near-uniform (large).
# The critical transition for this subset was 0.1..10 in the quick diagnostic,
# which gets ~9 points under the default 20-point log spacing.
# (length>=2 required by range(); the script guards against length=1.)
const delta_G_values = [0.0; 10 .^ range(-1, 4; length=max(delta_g_points, 2))]

# Windows: (name, start_row, opp_density). day d starts at row d*96+1.
# Day 0 = weekday (schools/offices active), day 5 = weekend (schools off,
# offices off — different demand pattern).
const WINDOWS = [
    ("day0_weekday", 1,   0.0),
    ("day5_weekend", 481, 0.0),
]

# Basin classification (relative to dec_avg, so it auto-calibrates to the
# synthetic data's price/consumption scale). Thresholds are fractions of
# |dec_avg|: benefit < 2% -> "A_none", < 8% -> "B_local", >= 8% -> "C_global".
const BASIN_NONE_PCT  = 0.02
const BASIN_LOCAL_PCT = 0.08

# Output files (incremental writes survive a crash).
const OUT_CSV = "results/sweep_delta_G_synthetic.csv"
const OUT_SUM = "results/sweep_delta_G_synthetic_summary.txt"

# ---------- Load data and helpers ----------
# Load all 30 synthetic buildings (672-row series), then slice subset + window.
println("Loading synthetic data for 30 buildings (672 steps)...")
all_buildings_full, energy_cost, energy_sale = MPC_load_from_CSV(30, 672, "synthetic_data")
opt = MPC_optimiser(energy_cost', energy_sale')
println("Loaded. num_repeats=$num_repeats  delta_G_points=$(length(delta_G_values))  windows=$(length(WINDOWS))")

# Slice window AND re-ID buildings to 1..N (works around the b.id indexing bug).
function slice_window(buildings, subset_ids, s, nsteps)
    out = Vector{MPC_Building}(undef, length(subset_ids))
    for (i, bid) in enumerate(subset_ids)
        b = buildings[bid]
        out[i] = MPC_Building(b.loc, b.pred_cons[s:s+nsteps-1, :], b.pred_prod[s:s+nsteps-1, :],
            b.act_cons[s:s+nsteps-1], b.act_prod[s:s+nsteps-1],
            b.max_storage, b.storage_max_flow, b.charge_eff, b.discharge_eff,
            i, zeros(nsteps))   # re-ID to position i (1..N)
    end
    return out
end

basin_of(benefit, dec_avg) = benefit < BASIN_NONE_PCT * abs(dec_avg) ? "A_none" : benefit < BASIN_LOCAL_PCT * abs(dec_avg) ? "B_local" : "C_global"

# ---------- Per-window decentralised baselines ----------
println("\nComputing decentralised baselines per window:")
dec_avg = Dict{String, Float64}()
for (wname, s, dens) in WINDOWS
    bwin = slice_window(all_buildings_full, SUBSET_IDS, s, N_STEPS)
    dec_cost = 0.0
    for b in bwin
        c, _, _ = optimise(opt, [b], num_ahead, true, receding_horizon)
        dec_cost += c
    end
    dec_avg[wname] = dec_cost / num_builds
    println("  $wname (start=$s, dens=$dens): dec_avg=$(round(dec_avg[wname], digits=2))")
end

# ---------- Sweep ----------
data = DataFrame(window=[], delta_G=[], repeat=[], average_cost=[], num_iters=[],
                 time=[], benefit_vs_dec=[], basin=[])

total_runs = length(WINDOWS) * length(delta_G_values) * num_repeats
t_start = time()

# Wrap in `let` so done_runs is in a hard scope (avoids the soft-scope
# ambiguity that breaks `done_runs += 1` inside nested for/try at top level).
let done_runs = 0
for (wname, s, dens) in WINDOWS
    println("\n=== Window $wname (start=$s, dens=$dens, dec_avg=$(round(dec_avg[wname], digits=2))) ===")
    for (delta_G_idx, delta_G) in enumerate(delta_G_values)
        for repeat in 1:num_repeats
            try
                # Seed depends only on repeat (controlled across delta_G):
                # same RNG start for each repeat number -> the only thing
                # varying across delta_G is the softmax temperature.
                # For delta_G=0 the seed is irrelevant (argmax, no sampling):
                # identical repeats confirm determinism.
                Random.seed!(repeat)

                bwin = slice_window(all_buildings_full, SUBSET_IDS, s, N_STEPS)
                # Reset battery state (coal_MPC mutates b.SoC in place; see
                # AGENTS.md and MPC_optimiser.jl:519,535). Stale SoC corrupts
                # the next repeat's MPC trajectory.
                for b in bwin
                    fill!(b.SoC, 0.0)
                end
                t1 = time()

                res, _, num_iters = coal_MPC((buildings, mcs, k, na, rh) ->
                    privacy_focussed_coals_with_delta(buildings, mcs, k, na, rh, delta_G),
                    bwin, max_coal_size, num_ahead)

                t2 = time()
                runtime = t2 - t1
                avg_cost = res / num_builds
                benefit = dec_avg[wname] - avg_cost   # positive = coalition cheaper
                push!(data, [wname, delta_G, repeat, avg_cost, num_iters, runtime, benefit, basin_of(benefit, dec_avg[wname])])

                CSV.write(OUT_CSV, data)
                done_runs += 1
                elapsed = time() - t_start
                eta = done_runs > 1 ? elapsed / done_runs * (total_runs - done_runs) : 0.0
                println("  [$wname] dg=$delta_G rep=$repeat cost=$(round(avg_cost, digits=1)) benefit=$(round(benefit, digits=1)) ($(round(100*benefit/abs(dec_avg[wname]), digits=2))%) basin=$(basin_of(benefit, dec_avg[wname])) | $(done_runs)/$total_runs ETA=$(round(eta/60, digits=1))min")
            catch e
                println("  FAILED [$wname] dg=$delta_G rep=$repeat: $(typeof(e)): $e")
            end
        end
    end
end
end # let done_runs

# ---------- Summary ----------
println("\n============================================================")
println("Summary: subset=$(SUBSET_IDS)  num_repeats=$num_repeats")
println("============================================================")
open(OUT_SUM, "w") do io
    println(io, "sweep_delta_G_synthetic summary")
    println(io, "subset_ids = $(SUBSET_IDS)")
    println(io, "num_repeats = $num_repeats")
    println(io, "delta_G_values = $(delta_G_values)")
    println(io, "windows = $(WINDOWS)")
    println(io, "dec_avg = $(dec_avg)")
    println(io)
    for (wname, s, dens) in WINDOWS
        println(io, "=== Window $wname (dens=$dens, dec_avg=$(dec_avg[wname])) ===")
        sub = filter(:window => ==(wname), data)
        if nrow(sub) == 0
            println(io, "  (no data)")
            continue
        end
        summ = combine(groupby(sub, :delta_G),
            :average_cost  => mean => :mean_cost,
            :average_cost  => std  => :std_cost,
            :average_cost  => minimum => :min_cost,
            :average_cost  => maximum => :max_cost,
            :benefit_vs_dec => mean => :mean_benefit,
            :benefit_vs_dec => std  => :std_benefit,
            :basin => (b -> count(==("A_none"), b))  => :n_A_none,
            :basin => (b -> count(==("B_local"), b)) => :n_B_local,
            :basin => (b -> count(==("C_global"), b)) => :n_C_global,
            :average_cost => length => :n_runs,
        )
        sort!(summ, :delta_G)
        println(io, summ)
        println(io)
    end
end

for (wname, s, dens) in WINDOWS
    println("\n--- Window $wname (dens=$dens, dec_avg=$(round(dec_avg[wname], digits=2))) ---")
    sub = filter(:window => ==(wname), data)
    summ = combine(groupby(sub, :delta_G),
        :average_cost  => mean => :mean_cost,
        :average_cost  => std  => :std_cost,
        :benefit_vs_dec => mean => :mean_benefit,
        :basin => (b -> count(==("A_none"), b))  => :n_A,
        :basin => (b -> count(==("B_local"), b)) => :n_B,
        :basin => (b -> count(==("C_global"), b)) => :n_C,
    )
    sort!(summ, :delta_G)
    println(summ)
end

# Determinism check: delta_G=0 repeats must be identical.
for (wname, s, dens) in WINDOWS
    dg0 = filter(r -> r.window == wname && r.delta_G == 0.0, data)
    if nrow(dg0) > 1
        spread = maximum(dg0.average_cost) - minimum(dg0.average_cost)
        if spread == 0.0
            println("Determinism check [$wname]: PASS (delta_G=0 identical across $(nrow(dg0)) repeats, cost=$(dg0.average_cost[1]))")
        else
            println("Determinism check [$wname]: FAIL (delta_G=0 spread=$spread across $(nrow(dg0)) repeats)")
        end
    end
end

println("\nResults saved:")
println("  $OUT_CSV  (per-run data)")
println("  $OUT_SUM  (per-delta_G summary table)")
