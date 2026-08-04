# Diagnostic: delta_G=0 (deterministic) vs several stochastic settings.
#
# Purpose: characterize whether variance in average_cost is monotonic in
# delta_G, and whether the spread comes from (a) a genuine bimodality that
# is present at *all* delta_G (basin-hopping driven by near-ties in
# poss_coal_vals, locked in by SoC propagation across the MPC horizon), or
# (b) a hidden nondeterminism bug (e.g. SCS/threading) that would make
# delta_G=0 itself non-identical across repeats.
#
# Key design choices vs experiments/private_delta_G_test.jl:
#   * delta_G=0.0 is included and exercises the argmax branch in
#     privacy_focussed_coals_with_delta (Coalition.jl:420). All 10 of its
#     repeats should be IDENTICAL -- that is the determinism check.
#   * Seed depends only on `repeat`, not on delta_G index, so each repeat
#     shares an RNG start across delta_G values (controlled comparison).
#   * A `basin` column tags each run as "low" (<450) or "high" (>=450) to
#     surface the bimodal split at a glance.
#   * SoC is reset between repeats (coal_MPC mutates b.SoC in place; see
#     AGENTS.md and MPC_optimiser.jl:519,535).
#
# Run from this directory:
#   julia --threads auto diag_delta_G_vs0.jl
# (scripts here use include("../...") -- see AGENTS.md.)

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
include("../plotting.jl")

num_builds = 10
max_coal_size = 6
num_steps = 96
num_ahead = 8
receding_horizon = false

# Load building data once; reuse across all runs (SoC reset between runs).
all_buildings, energy_cost, energy_sale = MPC_load_from_CSV(num_builds, num_steps)
opt = MPC_optimiser(energy_cost', energy_sale')

# 0.0 = deterministic (argmax). 0.1 ~ greedy. 10.0 ~ softmax.
# 100.0 ~ near-uniform (matches the legacy default in privacy_focussed_coals).
delta_G_values = [0.0, 0.1, 10.0, 100.0]
num_repeats = 10

# Basin threshold: from results/delta_G_0_1_eps.csv, costs cluster into
# ~405-415 ("low") and ~489-495 ("high"). 450 sits in the gap between them.
BASIN_THRESHOLD = 450.0

data = DataFrame(delta_G=[], repeat=[], average_cost=[], num_iters=[],
                 time=[], basin=[])

for (delta_G_idx, delta_G) in enumerate(delta_G_values)
    println("Testing delta_G = $delta_G")
    for repeat in 1:num_repeats
        try
            # Seed depends ONLY on repeat: same RNG start across delta_G
            # values -> controlled cross-delta_G comparison. For delta_G=0
            # the seed is irrelevant (no sampling) but is set anyway for
            # uniformity; identical results across repeats => deterministic.
            Random.seed!(repeat)

            buildings = all_buildings[1:num_builds]
            # Reset battery state: coal_MPC mutates b.SoC[k+1] in place.
            for b in buildings
                fill!(b.SoC, 0.0)
            end
            t1 = time()

            res, _, num_iters = coal_MPC((buildings, max_coal_size, k, num_ahead, receding_horizon) ->
                privacy_focussed_coals_with_delta(buildings, max_coal_size, k, num_ahead, receding_horizon, delta_G),
                buildings, max_coal_size, num_ahead)

            t2 = time()
            runtime = t2 - t1
            avg_cost = res / num_builds
            basin = avg_cost < BASIN_THRESHOLD ? "low" : "high"
            push!(data, [delta_G, repeat, avg_cost, num_iters, runtime, basin])

            CSV.write("results/diag_delta_G_vs0.csv", data)
            println("  repeat=$repeat cost=$(round(avg_cost, digits=3)) basin=$basin iters=$(round(num_iters, digits=2)) t=$(round(runtime, digits=1))s")
        catch e
            println("Failed at delta_G=$delta_G, repeat=$repeat")
            println(e)
        end
    end
end

println("\n=== Summary ===")
# Per delta_G: mean, std, count of low/high basin hits.
summary = combine(groupby(data, :delta_G),
    :average_cost => mean => :mean_cost,
    :average_cost => std  => :std_cost,
    :average_cost => length => :n_runs,
    :basin => (b -> count(==("low"), b))  => :n_low,
    :basin => (b -> count(==("high"), b)) => :n_high,
)
sort!(summary, :delta_G)
println(summary)

# Determinism check: delta_G=0.0 must give identical cost across repeats.
dg0 = filter(:delta_G => ==(0.0), data)
if nrow(dg0) > 0
    spread = maximum(dg0.average_cost) - minimum(dg0.average_cost)
    if spread == 0.0
        println("Determinism check: PASS (delta_G=0 identical across $(nrow(dg0)) repeats, cost=$(dg0.average_cost[1]))")
    else
        println("Determinism check: FAIL (delta_G=0 spread=$spread across $(nrow(dg0)) repeats)")
        println("  -> indicates hidden nondeterminism (SCS/threading). Inspect:")
        for r in sort(dg0, :repeat)
            println("     repeat=$(r.repeat) cost=$(r.average_cost)")
        end
    end
end

println("\nResults saved to results/diag_delta_G_vs0.csv")
