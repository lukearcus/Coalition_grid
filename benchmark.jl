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
#   2. Limited-info (non-DP)  — privacy_focussed_coals (deterministic greedy)
#   3. Full-info greedy       — bottom_up_full_info (OFF by default, slow)
#   4. Differential privacy   — privacy_focussed_coals_with_delta (OFF by default)
#
# Methods 3 and 4 are included for reference but disabled by default so
# collaborators who only need non-DP results can run this script as-is.
# Set RUN_FULL_INFO=true or RUN_DP=true at the top to enable them.

include("Buildings.jl")
include("MPC_optimiser.jl")
include("Coalition.jl")
include("load_EMS_data.jl")

using Printf
using Random

# ---- Configuration (edit these) ----
const NUM_BUILDS      = 8
const MAX_COAL        = 4          # 4 keeps ADMM solves tractable (6 is very slow)
const NUM_STEPS       = 48         # 12 hours (96 = full day; 48 includes k=40 where merges happen)
const NUM_LOOK_AHEAD  = 8          # MPC horizon
const RUN_FULL_INFO   = false      # full-info greedy (slow: solves all pairs)
const RUN_DP          = false      # DP comparison (off by default for collaborators)
const DP_DELTA_G      = 100.0      # privacy budget (only used if RUN_DP=true)
const DP_SEEDS        = [42, 1, 7] # seeds for the stochastic DP runs

# coal_MPC mutates building.SoC in place, so each method needs a fresh set of
# buildings.  This helper reloads from CSV, runs coal_MPC, and prints one row.
function run_method(label::String, coal_former)
    bs, _, _, _ = MPC_load_from_CSV(NUM_BUILDS, NUM_STEPS)
    t0 = time()
    cost, _, iters, _ = coal_MPC(coal_former, bs, MAX_COAL, NUM_LOOK_AHEAD)
    t1 = time()
    @printf "  %-32s  cost=%10.2f  time=%7.2fs  iters/step=%5.2f\n" label cost (t1 - t0) iters
    return cost
end

# A decentralised coal_former: every building stays a singleton (no merges).
function decentralised(bs, max_coal_size, k, num_look_ahead, receding_horizon=false)
    agents = Vector(1:length(bs))
    outs = [single_optimise_ADMM(opt, bs[a], k, num_look_ahead, receding_horizon) for a in agents]
    vars = [out[1] for out in outs]
    num_iters = sum([out[2] for out in outs])
    return agents, vars, num_iters
end

# ---- Shared setup ----
buildings, energy_cost, energy_sale, timestamps = MPC_load_from_CSV(NUM_BUILDS, NUM_STEPS)
global opt = MPC_optimiser(energy_cost', energy_sale')

println("=" ^ 72)
@printf "Benchmark: %d buildings, %d timesteps, horizon=%d, max_coal=%d\n" NUM_BUILDS NUM_STEPS NUM_LOOK_AHEAD MAX_COAL
println("=" ^ 72)
println()
println("Method                              cost          time     iters/step")
println("-" ^ 72)

# ---- Always-on methods ----
run_method("Decentralised (no coop)",    decentralised)
run_method("Limited-info (non-DP)",      privacy_focussed_coals)

# ---- Full-info greedy (optional, slow) ----
if RUN_FULL_INFO
    run_method("Full-info greedy",        bottom_up_full_info)
else
    println()
    println("  (Full-info greedy disabled — set RUN_FULL_INFO=true to enable)")
end

# ---- DP method (optional) ----
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
