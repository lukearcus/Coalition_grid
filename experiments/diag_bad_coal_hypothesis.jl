# =============================================================================
# Diagnostic: verify the "bad coalition sampling" hypothesis for the delta_G
# sweep on the synthetic subset.
#
# Observation (results/sweep_delta_G_synthetic.csv, day0 window): as delta_G
# grows, std(cost), benefit, ADMM iterations and runtime all DROP.
# Hypothesis: at high delta_G the (near-uniform) softmax more often samples
# low/zero-value pairs -- coalitions whose members barely trade. Such
# coalitions (a) cost little extra ADMM work (weak coupling -> fast
# convergence) and (b) do no worse than the members acting alone (no trade,
# plus the fair-price/dissolve IR floor in coal_MPC), so outcomes collapse
# toward the "mediocre" decentralised level: the good tail disappears and
# the spread shrinks.
#
# Part 1 (pair-level, no coalition sampling): for all 15 pairs of the subset:
#   - pair_val: the exponential-mechanism utility the sampler sees at k=1
#     (SoC=0), computed exactly as in privacy_focussed_coals_with_delta.
#   - harm: forced-pair full-MPC cost (optimise on the two buildings as one
#     coalition) minus the sum of the two members' singleton full-MPC costs.
#   - pair ADMM iterations per MPC step.
# Verifies: (i) how many pairs are zero/low value; (ii) harm ~ 0 for them
# ("don't do worse than working on their own"); (iii) low-value pairs need
# fewer ADMM iterations. NOTE: optimise bypasses coal_MPC's fair-price
# dissolve mechanism, so Part 1 measures the *pure joint-optimisation* harm.
#
# Part 2 (instrumented mini-sweep, day0 window): delta_G in
# {0, 0.359, 16.68, 1e4} x 10 repeats, same protocol as
# experiments/sweep_delta_G_subset.jl, with LOG_PICKS=true (runtime toggle in
# Coalition.jl) and PICK_LOG cleared before each run. Per run we record the
# usual cost/benefit/iters/time plus, from the pick log: n_picks, fraction of
# zero-value picks, fraction of "bad" picks (< 5% of that draw's pool argmax),
# mean picked/argmax ratio, total regret, fraction argmax, merges per step;
# and from coal_MPC extras: total internal trade volume and dissolve count.
#
# Caveat: Part 1 pair values are evaluated at k=1 (SoC=0). Later MPC steps
# see different pair values (SoC drift), so Part 1's zero-pair fraction is
# only an approximation of the sampling prior; Part 2 measures actual picks.
#
# Run from repo root:
#   julia --threads auto experiments/diag_bad_coal_hypothesis.jl
# (loaders read synthetic_data/ relative to cwd)
# =============================================================================

try
    using CSV
catch
    using CSV
end
@eval CSV.Parsers import Base.Ryu: writeshortest
using Combinatorics
using DataFrames
using Random
using Statistics
include("../Buildings.jl")
include("../MPC_optimiser.jl")
include("../Coalition.jl")
include("../load_EMS_data.jl")

# ---------- Configuration (mirrors sweep_delta_G_subset.jl) ----------
const SUBSET_IDS = [1, 10, 16, 25, 28, 7]
const num_builds = length(SUBSET_IDS)
const max_coal_size = 2
const num_ahead = 8
const receding_horizon = false
const N_STEPS = 96
const DAY0_START = 1
const WNAME = "day0_weekday"

# Same grid points as results/sweep_delta_G_synthetic.csv (deterministic
# anchor, near-greedy, near-uniform, uniform) so numbers are comparable.
const delta_G_values = [0.0, 0.35938136638046275, 16.68100537200059, 1e4]
const num_repeats = 10
const BAD_PICK_REL = 0.05   # pick is "bad" if < 5% of that draw's pool argmax
const ZERO_VAL = 1e-9      # pair value below this = "no trade" pair

const OUT_PAIR = "results/diag_pair_harm.csv"
const OUT_PICK = "results/diag_bad_coal_hypothesis.csv"
const OUT_SUM  = "results/diag_bad_coal_hypothesis_summary.txt"

# ---------- Load data and helpers ----------
println("Loading synthetic data for 30 buildings (672 steps)...")
all_buildings_full, energy_cost, energy_sale = MPC_load_from_CSV(30, 672, "synthetic_data")
opt = MPC_optimiser(energy_cost', energy_sale')
println("Loaded. day0 subset=$SUBSET_IDS  repeats=$num_repeats  dg=$(delta_G_values)")

# Slice window AND re-ID buildings to 1..N (works around the b.id indexing
# bug noted in sweep_delta_G_subset.jl: coal_MPC indexes prices by b.id).
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

# Fresh slice with zeroed SoC (coal_MPC / optimise mutate b.SoC in place;
# see AGENTS.md and MPC_optimiser.jl:519,535).
fresh_window() = begin
    bwin = slice_window(all_buildings_full, SUBSET_IDS, DAY0_START, N_STEPS)
    for b in bwin
        fill!(b.SoC, 0.0)
    end
    bwin
end

function pearson(x, y)
    n = length(x)
    n < 3 && return NaN
    mx, my = mean(x), mean(y)
    vx = sum((a - mx)^2 for a in x)
    vy = sum((b - my)^2 for b in y)
    (vx <= 0 || vy <= 0) && return NaN
    return sum((a - mx) * (b - my) for (a, b) in zip(x, y)) / sqrt(vx * vy)
end

# ==============================================================================
# Part 1: pair-level value / harm table
# ==============================================================================
println("\n=== Part 1: pair-level value / harm table (day0, k=1, horizon=$num_ahead) ===")

# Singleton net-consumption plans at k=1 with SoC=0 — exactly what
# privacy_focussed_coals_with_delta sees in round 1 of MPC step 1.
bwin = fresh_window()
cvs = Vector{Vector{Float64}}(undef, num_builds)
for j in 1:num_builds
    res = single_optimise_ADMM(opt, [bwin[j]], 1, num_ahead, receding_horizon)
    cvs[j] = vec(sum(value(res[1][2] - res[1][3]), dims=2))
end

# Pair utility, same arithmetic as Coalition.jl pair_val_mat (opp-sign min).
energy_diff = opt.energy_cost - opt.energy_sale
diff_vec = energy_diff[1:num_ahead]          # horizon = min(96-1+1, 8) = 8
max_price_diff = maximum(abs.(energy_diff))

function pair_value(cv_i, cv_j)
    v = 0.0
    for t in 1:num_ahead
        if cv_i[t] * cv_j[t] < 0
            v += diff_vec[t] * min(abs(cv_i[t]), abs(cv_j[t]))
        end
    end
    return v
end

# Softmax sampling probability of each pair at a given delta_G (epsilon=1 in
# Coalition.jl; delta_u = max(dg*max_price_diff, 1e-10)). Max-subtraction,
# same numerically-stable form as Coalition.jl.
function stable_probs(vals, dg)
    delta_u = max(dg * max_price_diff, 1e-10)
    ws = [v / (2 * delta_u) for v in vals]
    mw = maximum(ws)
    ps = [exp(w - mw) for w in ws]
    return ps ./ sum(ps)
end

pair_vals = Dict{Tuple{Int,Int},Float64}()
for (i, j) in combinations(1:num_builds, 2)
    pair_vals[(i, j)] = pair_value(cvs[i], cvs[j])
end
n_zero_pairs = count(v -> v < ZERO_VAL, values(pair_vals))
println("Pair values at k=1: $(n_zero_pairs)/$(length(pair_vals)) pairs are zero-value (< $ZERO_VAL)")

# Singleton full-MPC costs (decentralised baseline; same call as the sweep).
dec_cost = zeros(num_builds)
for j in 1:num_builds
    c, _, _ = optimise(opt, [fresh_window()[j]], num_ahead, true, receding_horizon)
    dec_cost[j] = c
end
dec_avg = sum(dec_cost) / num_builds
println("dec_avg (day0) = $(round(dec_avg, digits=2))  [sweep reference: 427.85]")

# Forced-pair full-MPC runs.
pair_data = DataFrame(i=Int[], j=Int[], ids=String[], pair_val=Float64[],
    p_dg0p36=Float64[], p_dg1e4=Float64[],
    pair_cost=Float64[], single_sum=Float64[], harm=Float64[], harm_pct=Float64[],
    iters_step=Float64[])

pair_list = collect(keys(pair_vals))
val_list = [pair_vals[k] for k in pair_list]
p36  = stable_probs(val_list, 0.35938136638046275)
p1e4 = stable_probs(val_list, 1e4)
p36_map  = Dict(pair_list[i] => p36[i] for i in eachindex(pair_list))
p1e4_map = Dict(pair_list[i] => p1e4[i] for i in eachindex(pair_list))

for (i, j) in combinations(1:num_builds, 2)
    bw = fresh_window()
    c_pair, _, it_pair = optimise(opt, bw[[i, j]], num_ahead, true, receding_horizon)
    v = pair_vals[(i, j)]
    ss = dec_cost[i] + dec_cost[j]
    push!(pair_data, (i, j, "$(SUBSET_IDS[i])+$(SUBSET_IDS[j])", v,
        p36_map[(i, j)], p1e4_map[(i, j)],
        c_pair, ss, c_pair - ss, 100 * (c_pair - ss) / ss, it_pair))
end
sort!(pair_data, :pair_val, rev=true)
CSV.write(OUT_PAIR, pair_data)
println("\nPair table (sorted by pair_val, harm>0 = pair worse than members alone):")
println(pair_data)

zero_rows = filter(:pair_val => v -> v < ZERO_VAL, pair_data)
pos_rows  = filter(:pair_val => v -> v >= ZERO_VAL, pair_data)
println("\nZero-value pairs ($(nrow(zero_rows))):")
println("  harm: mean=$(round(isempty(zero_rows) ? NaN : mean(zero_rows.harm), digits=3))  max=$(round(isempty(zero_rows) ? NaN : maximum(zero_rows.harm), digits=3))  |harm|%: max=$(round(isempty(zero_rows) ? NaN : maximum(abs.(zero_rows.harm_pct)), digits=3))")
println("  iters/step: mean=$(round(isempty(zero_rows) ? NaN : mean(zero_rows.iters_step), digits=1))")
println("Positive-value pairs ($(nrow(pos_rows)):")
println("  harm: mean=$(round(isempty(pos_rows) ? NaN : mean(pos_rows.harm), digits=3))  (negative = pair beats members alone)")
println("  iters/step: mean=$(round(isempty(pos_rows) ? NaN : mean(pos_rows.iters_step), digits=1))")
println("corr(pair_val, iters_step) over 15 pairs = $(round(pearson(pair_data.pair_val, pair_data.iters_step), digits=3))")
println("corr(pair_val, harm)       over 15 pairs = $(round(pearson(pair_data.pair_val, pair_data.harm), digits=3))")

# ==============================================================================
# Part 2: instrumented mini-sweep (day0), pick-level statistics
# ==============================================================================
println("\n=== Part 2: instrumented mini-sweep (day0, LOG_PICKS=true) ===")

LOG_PICKS = true   # runtime toggle on the global in Coalition.jl

pick_data = DataFrame(window=String[], delta_G=Float64[], repeat=Int[],
    average_cost=Float64[], benefit_vs_dec=Float64[], iters_per_step=Float64[],
    time=Float64[], internal_trade=Float64[], dissolves=Int[],
    n_picks=Int[], picks_per_k=Float64[], frac_zero_picks=Float64[],
    frac_bad_picks=Float64[], mean_pick_ratio=Float64[], mean_pick_val=Float64[],
    mean_argmax_val=Float64[], regret_total=Float64[], frac_argmax=Float64[],
    merges_per_k=Float64[])

total_runs = length(delta_G_values) * num_repeats
t_start = time()

let done_runs = 0
for delta_G in delta_G_values
    println("\n--- delta_G = $delta_G ---")
    for repeat in 1:num_repeats
        try
            # Seed depends only on repeat (same protocol as the main sweep:
            # controlled comparison across delta_G).
            Random.seed!(repeat)
            bwin = fresh_window()
            empty!(PICK_LOG)

            t1 = time()
            res, out2, iters_mean, stab, diss = coal_MPC(
                (buildings, mcs, k, na, rh) ->
                    privacy_focussed_coals_with_delta(buildings, mcs, k, na, rh, delta_G),
                bwin, max_coal_size, num_ahead)
            runtime = time() - t1

            avg_cost = res / num_builds
            benefit = dec_avg - avg_cost
            # coal_MPC returns [buy, sell, coal_exch_mat, internal_rev]; each
            # trade appears with opposite signs on the two members.
            internal_trade = sum(abs.(out2[3])) / 2

            # PICK_LOG entries:
            # (k, round, picked_ids, picked_val, argmax_ids, argmax_val, is_argmax, delta_G)
            n_picks = length(PICK_LOG)
            n_zero  = count(p -> p[4] < ZERO_VAL, PICK_LOG)
            n_bad   = count(p -> p[6] > ZERO_VAL && p[4] < BAD_PICK_REL * p[6], PICK_LOG)
            ratios  = [p[6] > ZERO_VAL ? p[4] / p[6] : 1.0 for p in PICK_LOG]
            mean_ratio  = isempty(ratios) ? NaN : mean(ratios)
            mean_pv     = n_picks > 0 ? mean(p[4] for p in PICK_LOG) : NaN
            mean_av     = n_picks > 0 ? mean(p[6] for p in PICK_LOG) : NaN
            regret     = sum(p[6] - p[4] for p in PICK_LOG)
            n_am       = count(p -> p[7], PICK_LOG)
            # max_coal_size=2: a drawn pair merges iff it has exactly 2 ids.
            n_merges   = count(p -> length(p[3]) == 2, PICK_LOG)

            push!(pick_data, (WNAME, delta_G, repeat, avg_cost, benefit, iters_mean,
                runtime, internal_trade, diss, n_picks, n_picks / N_STEPS,
                n_picks > 0 ? n_zero / n_picks : NaN,
                n_picks > 0 ? n_bad / n_picks : NaN,
                mean_ratio, mean_pv, mean_av, regret,
                n_picks > 0 ? n_am / n_picks : NaN,
                n_merges / N_STEPS))
            CSV.write(OUT_PICK, pick_data)

            done_runs += 1
            eta = done_runs > 1 ? (time() - t_start) / done_runs * (total_runs - done_runs) : 0.0
            println("  rep=$repeat cost=$(round(avg_cost, digits=1)) ben=$(round(benefit, digits=2)) iters=$(round(iters_mean, digits=1)) t=$(round(runtime, digits=1))s | picks=$n_picks zero=$(round(n_picks > 0 ? n_zero / n_picks : NaN, digits=2)) bad=$(round(n_picks > 0 ? n_bad / n_picks : NaN, digits=2)) ratio=$(round(mean_ratio, digits=2)) regret=$(round(regret, digits=1)) trade=$(round(internal_trade, digits=1)) diss=$diss | $(done_runs)/$total_runs ETA=$(round(eta / 60, digits=1))min")
        catch e
            println("  FAILED dg=$delta_G rep=$repeat: $(typeof(e)): $e")
        end
    end
end
end # let done_runs

LOG_PICKS = false   # restore

# ---------- Summary and verdicts ----------
println("\n============================================================")
println("Per-delta_G summary (day0, dec_avg=$(round(dec_avg, digits=2)))")
println("============================================================")
summ = combine(groupby(pick_data, :delta_G),
    :average_cost    => mean    => :mean_cost,
    :average_cost    => std     => :std_cost,
    :average_cost    => minimum => :min_cost,
    :average_cost    => maximum => :max_cost,
    :benefit_vs_dec  => mean    => :mean_benefit,
    :iters_per_step  => mean    => :mean_iters,
    :time            => mean    => :mean_time,
    :internal_trade  => mean    => :mean_trade,
    :dissolves       => mean    => :mean_diss,
    :frac_zero_picks => mean    => :mean_frac_zero,
    :frac_bad_picks  => mean    => :mean_frac_bad,
    :mean_pick_ratio => mean    => :mean_ratio,
    :regret_total    => mean    => :mean_regret,
    :merges_per_k    => mean    => :mean_merges_k,
    :n_picks         => length  => :n_runs,
)
sort!(summ, :delta_G)
println(summ)

open(OUT_SUM, "w") do io
    println(io, "diag_bad_coal_hypothesis summary")
    println(io, "subset_ids = $(SUBSET_IDS)")
    println(io, "window = $(WNAME) (start=$(DAY0_START))  dec_avg = $(dec_avg)")
    println(io, "delta_G_values = $(delta_G_values)  num_repeats = $(num_repeats)")
    println(io, "BAD_PICK_REL = $(BAD_PICK_REL)  ZERO_VAL = $(ZERO_VAL)")
    println(io)
    println(io, "--- Part 1: pair table ---")
    println(io, pair_data)
    println(io)
    println(io, "--- Part 2: per-delta_G summary ---")
    println(io, summ)
end

# Correlations across stochastic runs (exclude deterministic dg=0).
stoch = filter(:delta_G => dg -> dg > 0, pick_data)
println("\nAcross stochastic runs (n=$(nrow(stoch))):")
println("  corr(mean_pick_val, iters_per_step) = $(round(pearson(stoch.mean_pick_val, stoch.iters_per_step), digits=3))")
println("  corr(mean_pick_val, time)           = $(round(pearson(stoch.mean_pick_val, stoch.time), digits=3))")
println("  corr(internal_trade, iters_per_step)= $(round(pearson(stoch.internal_trade, stoch.iters_per_step), digits=3))")
println("  corr(internal_trade, benefit)       = $(round(pearson(stoch.internal_trade, stoch.benefit_vs_dec), digits=3))")
println("  corr(frac_bad_picks, benefit)       = $(round(pearson(stoch.frac_bad_picks, stoch.benefit_vs_dec), digits=3))")

# ---------- Verdicts ----------
println("\n============================================================")
println("VERDICTS")
println("============================================================")
println("V1 (bad coalitions sampled more at high delta_G):")
for r in eachrow(summ)
    println("  dg=$(r.delta_G): frac_zero=$(round(r.mean_frac_zero, digits=2))  frac_bad=$(round(r.mean_frac_bad, digits=2))  regret=$(round(r.mean_regret, digits=1))  [k=1 prior: $(n_zero_pairs)/15 = $(round(n_zero_pairs/15, digits=2)) zero-value]")
end
println("V2 (bad coalitions do no worse than solo): Part 1 zero-value pairs harm max = $(round(isempty(zero_rows) ? NaN : maximum(zero_rows.harm), digits=3)) (|pct| max = $(round(isempty(zero_rows) ? NaN : maximum(abs.(zero_rows.harm_pct)), digits=3))%)")
println("V3 (low value -> less ADMM work): Part 1 corr(pair_val, iters) = $(round(pearson(pair_data.pair_val, pair_data.iters_step), digits=3)); across runs corr(mean_pick_val, iters) = $(round(pearson(stoch.mean_pick_val, stoch.iters_per_step), digits=3))")
println("V4 (std drop = good tail disappearing): min_cost per dg: $(round.(summ.min_cost, digits=1))  max_cost per dg: $(round.(summ.max_cost, digits=1))")

# Determinism check: delta_G=0 repeats must be identical.
dg0 = filter(:delta_G => ==(0.0), pick_data)
if nrow(dg0) > 1
    spread = maximum(dg0.average_cost) - minimum(dg0.average_cost)
    println("\nDeterminism check [dg=0]: $(spread == 0.0 ? "PASS" : "FAIL") (spread=$spread across $(nrow(dg0)) repeats)")
end

println("\nResults saved:")
println("  $OUT_PAIR  (Part 1 pair table)")
println("  $OUT_PICK  (Part 2 per-run pick stats)")
println("  $OUT_SUM   (summary)")
