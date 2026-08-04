# delta_G sweep on the high-complementarity subset [62, 28, 70, 37, 18, 40].
#
# Background: Step 1 (see chat) showed this subset gives 2-4% coalition benefit
# vs 0.1% for buildings 1-10, and that delta_G=0 *beats* delta_G=10 here
# (opposite to buildings 1-10 where greedy is stuck in a local optimum and
# sampling helps). This sweep characterises the full curve.
#
# Uses day 37 (start_row=3553, opp-density 0.254 -- the best window for this
# subset) so coalitions have the most room to help.
#
# Note: buildings are re-IDed to 1..6 because coal_MPC indexes buy/sell by
# b.id assuming id==position (see MPC_optimiser.jl:525,527,540,542). The
# original IDs [62,28,70,37,18,40] would crash. This is a latent bug to fix
# separately.
#
# Run from repo root:
#   julia --threads auto experiments/diag_delta_G_subset.jl

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

# High-complementarity subset (3 PV producers + 3 net consumers).
# b62 (44% producer, 6.4 MW mean), b28/b70 (~42%, 0.2-0.9 MW),
# b37/b18/b40 (~1-3% producer, ~0.1 MW each).
SUBSET_IDS = [62, 28, 70, 37, 18, 40]
num_builds = length(SUBSET_IDS)
max_coal_size = 6
num_ahead = 8
receding_horizon = false

# Day 37 = best window for this subset (opp-density 0.254).
# cleaned_data rows are 96/day; day d starts at d*96+1.
START_ROW = 3553
N_STEPS = 96

# delta_G values: 0 (deterministic/greedy) -> 0.1 (sharp) -> 1 (soft) ->
# 10 (softmax) -> 100 (near-uniform). Spans the regimes seen in the
# buildings 1-10 sweep so we can compare the curves directly.
delta_G_values = [0.0, 0.1, 1.0, 10.0, 100.0]
num_repeats = 3

# Load full data for all 70 buildings, then slice the subset + window.
all_buildings_full, energy_cost, energy_sale = MPC_load_from_CSV(70, 4766)
opt = MPC_optimiser(energy_cost', energy_sale')

# Slice window AND re-ID buildings to 1..N (works around the b.id indexing
# bug in coal_MPC -- see header comment).
function slice_window(buildings, subset_ids, s, nsteps)
    out = Vector{MPC_Building}(undef, length(subset_ids))
    for (i, bid) in enumerate(subset_ids)
        b = buildings[bid]
        out[i] = MPC_Building(b.loc, b.pred_cons[s:s+nsteps-1, :], b.pred_prod[s:s+nsteps-1, :],
            b.act_cons[s:s+nsteps-1], b.act_prod[s:s+nsteps-1],
            b.max_storage, b.storage_max_flow, b.charge_eff, b.discharge_eff,
            i, zeros(nsteps))   # re-ID to position i (1..6)
    end
    return out
end

# Also compute the decentralised baseline (each building solved independently)
# for this subset+window, to anchor the coalition benefit.
let
    global dec_avg
    bwin = slice_window(all_buildings_full, SUBSET_IDS, START_ROW, N_STEPS)
    dec_cost = 0.0
    for b in bwin
        c, _, _ = optimise(opt, [b], num_ahead, true, receding_horizon)
        dec_cost += c
    end
    dec_avg = dec_cost / num_builds
end
println("Decentralised baseline (subset=$(SUBSET_IDS), day37): avg_cost=$(round(dec_avg, digits=2))")

data = DataFrame(delta_G=[], repeat=[], average_cost=[], num_iters=[], time=[], benefit_vs_dec=[])

for (delta_G_idx, delta_G) in enumerate(delta_G_values)
    println("Testing delta_G = $delta_G")
    for repeat in 1:num_repeats
        try
            # Seed depends only on repeat (controlled across delta_G).
            Random.seed!(repeat)

            bwin = slice_window(all_buildings_full, SUBSET_IDS, START_ROW, N_STEPS)
            # Reset battery state (coal_MPC mutates b.SoC in place).
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
            benefit = dec_avg - avg_cost   # positive = coalition cheaper
            push!(data, [delta_G, repeat, avg_cost, num_iters, runtime, benefit])

            CSV.write("results/diag_delta_G_subset.csv", data)
            println("  repeat=$repeat cost=$(round(avg_cost, digits=2)) benefit=$(round(benefit, digits=2)) ($(round(100*benefit/abs(dec_avg), digits=1))%) iters=$(round(num_iters, digits=1)) t=$(round(runtime, digits=1))s")
        catch e
            println("Failed at delta_G=$delta_G, repeat=$repeat")
            println(e)
        end
    end
end

println("\n=== Summary (day 37, subset=$(SUBSET_IDS), dec_avg=$(round(dec_avg, digits=2))) ===")
summary = combine(groupby(data, :delta_G),
    :average_cost => mean => :mean_cost,
    :average_cost => std  => :std_cost,
    :benefit_vs_dec => mean => :mean_benefit,
    :benefit_vs_dec => std  => :std_benefit,
    :average_cost => length => :n_runs,
)
sort!(summary, :delta_G)
println(summary)

println("\nResults saved to results/diag_delta_G_subset.csv")
