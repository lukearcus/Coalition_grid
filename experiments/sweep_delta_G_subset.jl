# =============================================================================
# delta_G sweep on a high-complementarity subset — REAL (EDF) or SYNTHETIC
# data, selected via the DATASET switch (see Configuration below).
#
# Real data: 70 buildings, 4766 steps (~49.6 days x 96 steps), prices are a
# 672-step buy/sell series (data/edf_prices.csv). Mean pairwise opposition
# density is ~11% overall, so subset/window choice matters far more than on
# the synthetic data (~39%).
#   Subset: small6 from experiments/search_subset_window.jl — PV producers
#   (62, 28, 70) mixed with net consumers (37, 18, 40).
#   Window: day37_best (start row 3553), the highest opposition-density
#   window (0.254) found by the earlier search. NOTE: window slicing re-bases
#   the series to k=1..96, so all windows use the same first-day price
#   pattern; only load/PV profiles differ (see search_subset_window.jl).
#
# Synthetic data (generate_synthetic_data.jl): 30 buildings across 10
# archetypes, 672 steps (7 days x 96), ~39% mean pairwise complementarity.
#   Subset: 3 staggered sellers (solar farm 1, coastal wind 25, hilltop wind
#   28) + 3 buyers with distinct peaks (industrial 10, data center 16,
#   residential 7).
#   Windows: day0_weekday + day5_weekend.
#
# ============ Preflight gate ============
# Before sweeping, the script computes all 15 pair utilities at k=1 (SoC=0)
# — the pool the exponential mechanism actually samples from — and ABORTS
# unless at least MIN_POSITIVE_PAIRS have positive value. Rationale: when
# all pair values are 0 the softmax is uniform for EVERY delta_G, so the
# sweep's delta_G axis is a no-op (this is exactly what happened with the
# synthetic-archetype subset [1,10,16,25,28,7] on real data: 15/15 pairs
# zero, 82/96 coalitions dissolved, benefit ~0.1-0.3% for all delta_G).
#
# ============ Extra columns ============
# Per run: dissolves (fair-price infeasible -> members enact singleton plans
# that step) and internal_trade (total |coalition exchange| volume / 2), so
# the "coalitions form but never trade / always dissolve" pathology is
# visible directly in the CSV. scs_fails counts SCS solves that terminated
# non-optimally (C-level LDL failures do not throw — see
# MPC_optimiser.jl SCS_FAIL_COUNT).
#
# ============ Configuration (edit here or set via ENV vars) ============
#   DATASET        (default "real")  "real" (EDF cleaned_data/) or
#                                   "synthetic" (synthetic_data/)
#   NUM_REPEATS    (default 10)  repeats per (window, delta_G)
#   DELTA_G_POINTS (default 10)  log-spaced points from 0.1 to 10000
#                                (plus delta_G=0 deterministic anchor)
#   OUT_CSV / OUT_SUM             output paths (defaults per dataset; override
#                                 for smoke tests so result files survive)
#
# ============ Known issue (worked around) ============
# coal_MPC indexes buy/sell by b.id assuming id==position. Non-contiguous
# subset IDs would crash, so slice_window re-IDs buildings to 1..N. Fix the
# indexing bug separately and this workaround can be dropped.
#
# ============ Run ============
#   julia --threads auto experiments/sweep_delta_G_subset.jl              # real
#   DATASET=synthetic julia --threads auto experiments/sweep_delta_G_subset.jl
# (from repo root -- loaders read cleaned_data//synthetic_data/ relative to cwd)
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
# Dataset switch: "real" (default) or "synthetic". Override per run:
#   DATASET=synthetic julia --threads auto experiments/sweep_delta_G_subset.jl
const DATASET = lowercase(get(ENV, "DATASET", "real"))

if DATASET == "real"
    # Real (EDF) data. Subset "small6" from search_subset_window.jl;
    # window day37_best = highest opposition-density window found (0.254).
    cfg = (label = "real (small6 @ day37_best)",
           load = (70, 4766, ""),
           subset = [62, 28, 70, 37, 18, 40],
           windows = [("day37_best", 3553, 0.254)],
           out_csv = "results/sweep_delta_G.csv",
           out_sum = "results/sweep_delta_G_summary.txt")
elseif DATASET == "synthetic"
    # Synthetic data. Subset: staggered sellers + distinct-peak buyers; two
    # windows (day 0 weekday, day 5 weekend).
    cfg = (label = "synthetic ([1,10,16,25,28,7], day0+day5)",
           load = (30, 672, "synthetic_data"),
           subset = [1, 10, 16, 25, 28, 7],
           windows = [("day0_weekday", 1, 0.0), ("day5_weekend", 481, 0.0)],
           out_csv = "results/sweep_delta_G_synthetic.csv",
           out_sum = "results/sweep_delta_G_synthetic_summary.txt")
else
    error("Unknown DATASET \"$DATASET\" (use \"real\" or \"synthetic\")")
end

# Derived config. OUT_CSV/OUT_SUM can be overridden (e.g. smoke tests
# writing to /tmp so the committed result files are not clobbered).
const SUBSET_IDS = cfg.subset
const WINDOWS    = cfg.windows
const OUT_CSV = get(ENV, "OUT_CSV", cfg.out_csv)
const OUT_SUM = get(ENV, "OUT_SUM", cfg.out_sum)

const num_builds = length(SUBSET_IDS)
const max_coal_size = 2
const num_ahead = 8
const receding_horizon = false
const N_STEPS = 96

# Preflight gate: minimum number of positive-value pairs (of 15) required
# at k=1, or the sweep aborts. See header rationale.
const MIN_POSITIVE_PAIRS = 4

# ENV overrides with sensible defaults.
const num_repeats    = parse(Int, get(ENV, "NUM_REPEATS", "10"))
const delta_g_points = parse(Int, get(ENV, "DELTA_G_POINTS", "10"))

# delta_G grid: 0.0 (deterministic argmax anchor) + log-spaced 0.01..1000.
# 0.01..1000 spans greedy (small) -> softmax (~1-10) -> near-uniform (large).
# (length>=2 required by range(); the script guards against length=1.)
const delta_G_values = [0.0; 10 .^ range(-1, 4; length=max(delta_g_points, 2))]

# Basin classification (relative to dec_avg, so it auto-calibrates to the
# active dataset's price/consumption scale). Thresholds are fractions of
# |dec_avg|: benefit < 2% -> "A_none", < 8% -> "B_local", >= 8% -> "C_global".
const BASIN_NONE_PCT  = 0.02
const BASIN_LOCAL_PCT = 0.08

# ---------- Load data and helpers ----------
# Load the full series for the active dataset, then slice subset + window.
println("Loading $DATASET data ($(cfg.load[1]) buildings, $(cfg.load[2]) steps)...")
all_buildings_full, energy_cost, energy_sale = MPC_load_from_CSV(cfg.load...)
opt = MPC_optimiser(energy_cost', energy_sale')
println("Loaded. dataset=$DATASET ($(cfg.label))  num_repeats=$num_repeats  delta_G_points=$(length(delta_G_values))  windows=$(length(WINDOWS))")

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

# ---------- Preflight gate: pair utilities at k=1 ----------
# Compute all 15 pair utilities exactly as privacy_focussed_coals_with_delta
# does in round 1 of MPC step 1 (SoC=0): singleton net-consumption plans,
# then the opp-sign min value over the horizon. If too few pairs have
# positive value, the softmax is (near-)uniform for every delta_G and the
# sweep would be uninformative — abort instead of burning ~1h of compute.
function preflight_pair_values(bwin)
    cvs = Vector{Vector{Float64}}(undef, length(bwin))
    for j in eachindex(bwin)
        res = single_optimise_ADMM(opt, [bwin[j]], 1, num_ahead, receding_horizon)
        cvs[j] = vec(sum(value(res[1][2] - res[1][3]), dims=2))
    end
    horizon = min(length(bwin[1].act_cons) - 1 + 1, num_ahead)
    diff_vec = (opt.energy_cost - opt.energy_sale)[1:horizon]
    vals = Dict{Tuple{Int,Int},Float64}()
    for i in 1:length(bwin), j in i+1:length(bwin)
        v = 0.0
        for t in 1:horizon
            if cvs[i][t] * cvs[j][t] < 0
                v += diff_vec[t] * min(abs(cvs[i][t]), abs(cvs[j][t]))
            end
        end
        vals[(i, j)] = v
    end
    return vals
end

println("\n=== Preflight: pair utilities at k=1 (SoC=0) per window ===")
gate_ok = true
for (wname, s, dens) in WINDOWS
    bwin = slice_window(all_buildings_full, SUBSET_IDS, s, N_STEPS)
    for b in bwin
        fill!(b.SoC, 0.0)
    end
    vals = preflight_pair_values(bwin)
    println("  [$wname] (start=$s, dens=$dens):")
    for (pair, v) in sort(collect(vals); by=last, rev=true)
        println("    ($(SUBSET_IDS[pair[1]]), $(SUBSET_IDS[pair[2]])) = $(round(v, digits=2))")
    end
    n_pos = count(v -> v > 1e-6, values(vals))
    println("    positive pairs: $n_pos/15 (gate requires >= $MIN_POSITIVE_PAIRS)")
    if n_pos < MIN_POSITIVE_PAIRS
        println("    *** PREFLIGHT GATE FAILED for $wname: the exponential-mechanism pool has too few positive-value pairs. Choose a different subset/window (see experiments/search_subset_window.jl). Aborting before the sweep. ***")
        global gate_ok = false
    end
end
if !gate_ok
    exit(1)
end

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
data = DataFrame(window=String[], delta_G=Float64[], repeat=Int[], average_cost=Float64[],
                num_iters=Float64[], time=Float64[], benefit_vs_dec=Float64[], basin=String[],
                dissolves=Int[], internal_trade=Float64[], scs_fails=Int[])

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

                # Reset the SCS failure counter (MPC_optimiser.jl) for this run.
                global SCS_FAIL_COUNT = 0

                res, out2, num_iters, stab, dissolves = coal_MPC((buildings, mcs, k, na, rh) ->
                    privacy_focussed_coals_with_delta(buildings, mcs, k, na, rh, delta_G),
                    bwin, max_coal_size, num_ahead)

                t2 = time()
                runtime = t2 - t1
                avg_cost = res / num_builds
                benefit = dec_avg[wname] - avg_cost   # positive = coalition cheaper
                # out2 = [buy, sell, coal_exch_mat, internal_rev]; each trade
                # appears with opposite signs on the two members, so /2 gives
                # the traded volume.
                internal_trade = sum(abs.(out2[3])) / 2
                push!(data, [wname, delta_G, repeat, avg_cost, num_iters, runtime, benefit, basin_of(benefit, dec_avg[wname]), dissolves, internal_trade, SCS_FAIL_COUNT])

                CSV.write(OUT_CSV, data)
                done_runs += 1
                elapsed = time() - t_start
                eta = done_runs > 1 ? elapsed / done_runs * (total_runs - done_runs) : 0.0
                println("  [$wname] dg=$delta_G rep=$repeat cost=$(round(avg_cost, digits=1)) benefit=$(round(benefit, digits=1)) ($(round(100*benefit/abs(dec_avg[wname]), digits=2))%) basin=$(basin_of(benefit, dec_avg[wname])) diss=$dissolves trade=$(round(internal_trade, digits=1)) scs_fails=$SCS_FAIL_COUNT | $(done_runs)/$total_runs ETA=$(round(eta/60, digits=1))min")
            catch e
                println("  FAILED [$wname] dg=$delta_G rep=$repeat: $(typeof(e)): $e")
            end
        end
    end
end
end # let done_runs

# ---------- Summary ----------
println("\n============================================================")
println("Summary: dataset=$DATASET ($(cfg.label))  subset=$(SUBSET_IDS)  num_repeats=$num_repeats")
println("============================================================")
open(OUT_SUM, "w") do io
    println(io, "sweep_delta_G summary (dataset=$DATASET: $(cfg.label))")
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
            :dissolves      => mean => :mean_dissolves,
            :internal_trade => mean => :mean_trade,
            :scs_fails      => mean => :mean_scs_fails,
            :num_iters      => mean => :mean_iters,
            :time           => mean => :mean_time,
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
        :dissolves      => mean => :mean_diss,
        :internal_trade => mean => :mean_trade,
        :scs_fails      => mean => :mean_scs_fails,
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
