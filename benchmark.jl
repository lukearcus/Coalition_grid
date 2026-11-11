# benchmark.jl — Coalition formation method comparison.
#
# Run:  OMP_NUM_THREADS=1 julia benchmark.jl
#
# OMP_NUM_THREADS=1 is required: SCS uses OpenMP which is slow for these small
# problems and deadlocks with Julia threads. The env var must be set before
# Julia starts (libgomp reads it at process load).
#
# Compares coalition-formation strategies on the same receding-horizon MPC
# problem (same buildings, prices, horizon):
#
#   1. Decentralised         — each building optimises alone (no cooperation)
#   2. Centralised           — grand coalition (one ADMM solve over all buildings)
#   3. Limited-info (non-DP) — privacy_focussed_coals (deterministic greedy)
#   4. Full-info greedy      — bottom_up_full_info (optional, slow)
#   5. Differential privacy  — privacy_focussed_coals_with_delta (optional)
#
# Methods 4 and 5 are gated by RUN_FULL_INFO / RUN_DP at the top.
#
# Deterministic methods (1-3, and full-info when enabled) are repeated
# NUM_REPS times to capture timing noise (SCS solve time varies run-to-run).
# Reported time is mean ± sample std; cost is deterministic across reps —
# a warning is printed if it drifts (solver nondeterminism red flag).
# DP timing/cost variation comes from DP_SEEDS, so DP is not repeated.

include("Buildings.jl")
include("MPC_optimiser.jl")
include("Coalition.jl")
include("load_EMS_data.jl")

using Printf
using Random
using Statistics

# ---- Configuration (edit these) ----
const NUM_BUILDS      = 8
const MAX_COAL        = 4          # 4 keeps ADMM solves tractable (6 is very slow)
const NUM_STEPS       = 48         # 12 hours (96 = full day; 48 includes k=40 where merges happen)
const NUM_LOOK_AHEAD  = 8          # MPC horizon
const NUM_REPS        = 3          # repeats for deterministic methods (timing noise)
const RUN_FULL_INFO   = true      # full-info greedy (slow: solves all pairs)
const RUN_DP          = true      # DP comparison (off by default for collaborators)
const DP_DELTA_G      = 100.0      # privacy budget (only used if RUN_DP=true)
const DP_SEEDS        = [42, 1, 7] # seeds for the stochastic DP runs

# coal_MPC mutates building.SoC in place, so each rep needs a fresh set of
# buildings (reloaded from CSV). Repeats NUM_REPS times and reports
# mean ± sample std of timing across reps. cost is deterministic across reps
# for these methods; a warning is printed if it drifts (solver nondeterminism
# red flag). coal_MPC returns 5 values; the 4th (stab_score) and 5th
# (dissolve_count) are dropped by the 4-way destructure below.
function run_method(label::String, coal_former)
    costs = Float64[]
    times = Float64[]
    iters_vals = Float64[]
    for rep in 1:NUM_REPS
        bs, _, _, _ = MPC_load_from_CSV(NUM_BUILDS, NUM_STEPS)
        t0 = time()
        cost, _, iters, _ = coal_MPC(coal_former, bs, MAX_COAL, NUM_LOOK_AHEAD)
        t1 = time()
        push!(costs, cost)
        push!(times, t1 - t0)
        push!(iters_vals, iters)
    end
    mean_cost  = mean(costs)
    mean_time  = mean(times)
    mean_iters = mean(iters_vals)
    std_time   = length(times) > 1 ? std(times) : 0.0
    if std(costs) > 1e-6
        @printf "  WARNING: %s cost drifted across reps (std=%.4e); solver may be nondeterministic\n" label std(costs)
    end
    @printf "  %-32s  cost=%10.2f  time=%7.2fs ± %5.2fs  iters/step=%6.2f  (n=%d)\n" label mean_cost mean_time std_time mean_iters NUM_REPS
    return mean_cost
end

# A decentralised coal_former: every building stays a singleton (no merges).
function decentralised(bs, max_coal_size, k, num_look_ahead, receding_horizon=false)
    agents = Vector(1:length(bs))
    outs = [single_optimise_ADMM(opt, bs[a], k, num_look_ahead, receding_horizon) for a in agents]
    vars = [out[1] for out in outs]
    num_iters = sum([out[2] for out in outs])
    return agents, vars, num_iters
end

# A centralised coal_former: every building in one grand coalition (the
# centralised baseline). Solved per-step via ADMM — the same solver as the
# other methods, so timing is directly comparable, and it flows through
# coal_MPC's SoC / cost pipeline. max_coal_size is ignored (the grand
# coalition is intentionally unconstrained). A grand coalition of NUM_BUILDS
# may approach the 1000-iter ADMM cap; the poss_sol fallback still yields a
# result. Returns 3 values (no singleton_vars), so coal_MPC skips the
# fair-price/dissolve path — there is no internal trade dispute in a single
# coalition.
function centralised(bs, max_coal_size, k, num_look_ahead, receding_horizon=false)
    states, num_iters = single_optimise_ADMM(opt, bs, k, num_look_ahead, receding_horizon)
    return [collect(1:length(bs))], [states], num_iters
end

# ---- Shared setup ----
buildings, energy_cost, energy_sale, timestamps = MPC_load_from_CSV(NUM_BUILDS, NUM_STEPS)
global opt = MPC_optimiser(energy_cost', energy_sale')

println("=" ^ 72)
@printf "Benchmark: %d buildings, %d timesteps, horizon=%d, max_coal=%d\n" NUM_BUILDS NUM_STEPS NUM_LOOK_AHEAD MAX_COAL
println("=" ^ 72)
println()
println("Method                              cost          time (mean ± std)   iters/step")
println("-" ^ 72)

# ---- Always-on methods ----
run_method("Decentralised (no coop)",    decentralised)
run_method("Centralised (grand coalition)", centralised)
run_method("Limited-info (non-DP)",      privacy_focussed_coals)

# ---- Full-info greedy (optional, slow) ----
if RUN_FULL_INFO
    run_method("Full-info greedy",        bottom_up_full_info)
else
    println()
    println("  (Full-info greedy disabled — set RUN_FULL_INFO=true to enable)")
end

# ---- DP method (optional) ----
# NUM_REPS does not apply here: DP timing/cost variation comes from the RNG
# draws (one run per seed in DP_SEEDS), not from SCS timing noise.
if RUN_DP
    for s in DP_SEEDS
        Random.seed!(s)
        run_method("DP (delta_G=$(DP_DELTA_G), seed=$(s))",
                   (bs, mcs, k, na, rh) ->
                       privacy_focussed_coals_with_delta(bs, mcs, k, na, rh, DP_DELTA_G))
    end
else
    println("  (DP comparison disabled — set RUN_DP=true to enable)")
end

# ---- Show coalition structure at k=40 for the non-DP method (merges happen there) ----
println()
println("Coalition structure at k=40 (non-DP):")
bs2, _, _, _ = MPC_load_from_CSV(NUM_BUILDS, NUM_STEPS)
coal, _, _ = privacy_focussed_coals(bs2, MAX_COAL, 40, NUM_LOOK_AHEAD)
for agent in coal
    println("  ", sort(vcat([a for a in agent]...)))
end
