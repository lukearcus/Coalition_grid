# Search for subsets/windows where delta_G=0 (deterministic greedy) shows
# coalition benefit over decentralised, using the current code (tight SCS 1e-4,
# argmax for delta_G=0, myopic split check).
#
# Tests 4 candidate subsets (6-15 buildings) × 3 windows (best/mid/worst
# complementarity). For each: decentralised baseline + delta_G=0 coalition
# (1 repeat -- deterministic). Reports benefit per subset/window.
#
# Run from repo root:
#   julia --threads auto experiments/search_subset_window.jl

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
# Candidate subsets: increasing size, mixing PV producers (b62/b70/b28/b31/b27)
# with net consumers (b37/b18/b40/b21/b50) and extra diversity (b17/b43/b11/b48/b63).
# max_coal_size grows with subset so larger coalitions can form.
const SUBSETS = [
    ("small6",     [62, 28, 70, 37, 18, 40],                                   6),
    ("balanced10", [62, 70, 28, 31, 27, 18, 40, 37, 21, 50],                   6),
    ("mixed12",    [62, 70, 28, 31, 27, 18, 40, 37, 21, 50, 17, 43],           8),
    ("big15",      [62, 70, 28, 31, 27, 18, 40, 37, 21, 50, 17, 43, 11, 48, 63], 8),
]

# Windows: (name, start_row, opp_density from earlier analysis).
# All windows start at 01:00 so prices are identical; only load/PV profiles differ.
const WINDOWS = [
    ("day37_best", 3553, 0.254),
    ("day43_mid",  4129, 0.242),
    ("day0_worst", 1,    0.148),
]

const num_ahead = 8
const receding_horizon = false
const N_STEPS = 96

# ---------- Load data ----------
println("Loading data for 70 buildings (full series)...")
all_buildings_full, energy_cost, energy_sale = MPC_load_from_CSV(70, 4766)
opt = MPC_optimiser(energy_cost', energy_sale')
println("Loaded.")

# Slice window AND re-ID buildings to 1..N (works around the b.id indexing
# bug in coal_MPC -- see MPC_optimiser.jl:525,527,540,542).
function slice_window(buildings, subset_ids, s, nsteps)
    out = Vector{MPC_Building}(undef, length(subset_ids))
    for (i, bid) in enumerate(subset_ids)
        b = buildings[bid]
        out[i] = MPC_Building(b.loc, b.pred_cons[s:s+nsteps-1, :], b.pred_prod[s:s+nsteps-1, :],
            b.act_cons[s:s+nsteps-1], b.act_prod[s:s+nsteps-1],
            b.max_storage, b.storage_max_flow, b.charge_eff, b.discharge_eff,
            i, zeros(nsteps))
    end
    return out
end

# ---------- Search ----------
data = DataFrame(subset=[], window=[], num_builds=[], max_coal_size=[],
                 dec_avg=[], coal_dg0_avg=[], benefit=[], benefit_pct=[])

for (sname, subset_ids, mcs) in SUBSETS
    num_builds = length(subset_ids)
    for (wname, s, dens) in WINDOWS
        try
            bwin = slice_window(all_buildings_full, subset_ids, s, N_STEPS)
            # Decentralised baseline
            dec_cost = 0.0
            for b in bwin
                c, _, _ = optimise(opt, [b], num_ahead, true, receding_horizon)
                dec_cost += c
            end
            dec_avg = dec_cost / num_builds
            # delta_G=0 coalition (deterministic, 1 repeat)
            for b in bwin; fill!(b.SoC, 0.0); end
            Random.seed!(1)
            coal_cost, _, _ = coal_MPC((buildings, maxcs, k, na, rh) ->
                privacy_focussed_coals_with_delta(buildings, maxcs, k, na, rh, 0.0),
                bwin, mcs, num_ahead)
            coal_avg = coal_cost / num_builds
            benefit = dec_avg - coal_avg
            pct = 100 * benefit / abs(dec_avg)
            push!(data, [sname, wname, num_builds, mcs, dec_avg, coal_avg, benefit, pct])
            println("[$sname | $wname] dec=$(round(dec_avg, digits=1)) coal_dg0=$(round(coal_avg, digits=1)) benefit=$(round(benefit, digits=1)) ($(round(pct, digits=2))%)")
            flush(stdout)
        catch e
            println("FAILED [$sname | $wname]: $(typeof(e)): $e")
            push!(data, [sname, wname, length(subset_ids), mcs, NaN, NaN, NaN, NaN])
        end
    end
end

println("\n=== Summary ===")
println(data)
println("\nResults saved to results/search_subset_window.csv")
CSV.write("results/search_subset_window.csv", data)
