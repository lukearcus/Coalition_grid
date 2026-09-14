# =============================================================================
# Lying-agent experiment: how does a strategic liar degrade honest agents'
# costs across delta_G values and lie levels?
#
# Designated liar (agent 1) fabricates its predicted consumption (pred_cons) to
# always appear as the perfect complement to the biggest seller in the pool,
# manipulating coalition formation to secure favourable internal trades.
# Three lie levels:
#   truthful    — no lie (control, matches sweep_delta_G_subset numbers)
#   bounded     — shift toward fabrication target, clipped to ±LIE_BOUND
#   fabrication — claim exact opposite net demand of biggest seller each step
#
# The liar's act_cons/act_prod/battery params stay truthful, so realised costs
# reflect reality while coalition/trade decisions are biased by the lie.
#
# ============ Configuration (edit here or set via ENV vars) ============
#   NUM_REPEATS    (default 5)
#   DELTA_G_POINTS (default 6)
#
# Runtime estimate (day 37 only, 6 buildings, ~80s/run):
#   (1 + DELTA_G_POINTS) * 3 * NUM_REPEATS * 80s
#   = 7 * 3 * 5 * 80s ~= 2.3 hours
#
# ============ Run ============
#   julia --threads auto experiments/sweep_lying_agent.jl
# (from repo root -- loaders read data/ and cleaned_data/ relative to cwd)
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
const SUBSET_IDS = [62, 28, 70, 37, 18, 40]
const num_builds = length(SUBSET_IDS)
const max_coal_size = 6
const num_ahead = 8
const receding_horizon = false
const N_STEPS = 96

const num_repeats    = parse(Int, get(ENV, "NUM_REPEATS", "5"))
const delta_g_points = parse(Int, get(ENV, "DELTA_G_POINTS", "6"))

const delta_G_values = [0.0; 10 .^ range(-1, 4; length=max(delta_g_points, 2))]

const LIE_LEVELS = ["truthful", "bounded", "fabrication"]
const LIAR_IDX = 1

# Bound for the bounded lie: half the total swept delta_G range (fixed).
const LIE_BOUND = (maximum(delta_G_values) - minimum(delta_G_values)) / 2

const WINDOWS = [
    ("day37_best", 3553, 0.254),
]

const BASIN_NONE  = 50.0
const BASIN_LOCAL = 250.0

const OUT_CSV = "results/sweep_lying_agent.csv"
const OUT_SUM = "results/sweep_lying_agent_summary.txt"

# ---------- Load data ----------
println("Loading data for 70 buildings (full series)...")
all_buildings_full, energy_cost, energy_sale = MPC_load_from_CSV(70, 4766)
opt = MPC_optimiser(energy_cost', energy_sale')
println("Loaded. num_repeats=$num_repeats  delta_G_points=$(length(delta_G_values))  lie_levels=$(length(LIE_LEVELS))  bound=$LIE_BOUND")

# ---------- Helpers ----------

# Slice window AND re-ID buildings to 1..N (works around the b.id indexing bug).
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

basin_of(benefit) = benefit < BASIN_NONE ? "A_none" : benefit < BASIN_LOCAL ? "B_local" : "C_global"

# Replace buildings[liar_idx] with a lying version. Returns clip_fraction.
function make_liar!(buildings::Vector{MPC_Building}, liar_idx::Int, lie_level::String, bound::Float64)
    lie_level == "truthful" && return 0.0

    liar = buildings[liar_idx]
    truth_nd = liar.pred_cons .- liar.pred_prod

    # Most negative net demand across other agents at each (k, h)
    other_nd = [buildings[j].pred_cons .- buildings[j].pred_prod
                for j in 1:length(buildings) if j != liar_idx]
    min_nd = reduce((a, b) -> min.(a, b), other_nd)

    # Fabrication: exact opposite of the biggest seller
    fab_nd = .-min_nd

    clip_fraction = 0.0
    if lie_level == "fabrication"
        lie_nd = fab_nd
    elseif lie_level == "bounded"
        shift = fab_nd .- truth_nd
        active = abs.(shift) .> bound
        clip_fraction = sum(active) / length(active)
        lie_nd = truth_nd .+ clamp.(shift, -bound, bound)
    else
        error("Unknown lie_level: $lie_level")
    end

    lie_pred_cons = liar.pred_prod .+ lie_nd

    buildings[liar_idx] = MPC_Building(
        liar.loc, lie_pred_cons, liar.pred_prod,
        liar.act_cons, liar.act_prod,
        liar.max_storage, liar.storage_max_flow, liar.charge_eff, liar.discharge_eff,
        liar.id, copy(liar.SoC))

    return clip_fraction
end

# Per-agent cost from the buy/sell matrices returned by coal_MPC.
function per_agent_costs(buy_mat, sell_mat, opt, n_steps)
    if n_steps <= 96
        buy_cost = opt.energy_cost[1:n_steps]
        sell_price = opt.energy_sale[1:n_steps]
    else
        buy_cost = hcat(repeat(opt.energy_cost, n_steps÷96), opt.energy_cost[1:(n_steps%96)])'
        sell_price = hcat(repeat(opt.energy_sale, n_steps÷96), opt.energy_sale[1:(n_steps%96)])'
    end
    return vec(buy_cost' * buy_mat - sell_price' * sell_mat)
end

# ---------- Decentralised baselines ----------
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
data = DataFrame(window=String[], delta_G=Float64[], lie_level=String[], repeat=Int[],
                 honest_avg_cost=Float64[], liar_cost=Float64[], total_avg_cost=Float64[],
                 num_iters=Float64[], time=Float64[], benefit_vs_dec=Float64[],
                 basin=String[], clip_fraction=Float64[])

total_runs = length(WINDOWS) * length(delta_G_values) * length(LIE_LEVELS) * num_repeats
t_start = time()

let done_runs = 0
for (wname, s, dens) in WINDOWS
    println("\n=== Window $wname (start=$s, dens=$dens, dec_avg=$(round(dec_avg[wname], digits=2))) ===")
    for delta_G in delta_G_values
        for lie_level in LIE_LEVELS
            for repeat in 1:num_repeats
                try
                    Random.seed!(repeat)

                    bwin = slice_window(all_buildings_full, SUBSET_IDS, s, N_STEPS)
                    clip_frac = make_liar!(bwin, LIAR_IDX, lie_level, LIE_BOUND)
                    for b in bwin
                        fill!(b.SoC, 0.0)
                    end
                    t1 = time()

                    res, trades, num_iters = coal_MPC((buildings, mcs, k, na, rh) ->
                        privacy_focused_coals_with_delta(buildings, mcs, k, na, rh, delta_G),
                        bwin, max_coal_size, num_ahead)

                    t2 = time()
                    runtime = t2 - t1

                    buy_mat, sell_mat = trades
                    pac = per_agent_costs(buy_mat, sell_mat, opt, N_STEPS)
                    honest_avg = mean(pac[2:end])
                    liar_cost = pac[1]
                    total_avg = res / num_builds
                    benefit = dec_avg[wname] - total_avg

                    push!(data, [wname, delta_G, lie_level, repeat,
                                honest_avg, liar_cost, total_avg,
                                num_iters, runtime, benefit, basin_of(benefit), clip_frac])
                    CSV.write(OUT_CSV, data)

                    done_runs += 1
                    elapsed = time() - t_start
                    eta = done_runs > 1 ? elapsed / done_runs * (total_runs - done_runs) : 0.0
                    println("  [$wname] dg=$delta_G lie=$lie_level rep=$repeat honest=$(round(honest_avg, digits=1)) liar=$(round(liar_cost, digits=1)) total=$(round(total_avg, digits=1)) benefit=$(round(benefit, digits=1)) ($(round(100*benefit/abs(dec_avg[wname]), digits=2))%) clip=$(round(clip_frac, digits=2)) | $(done_runs)/$total_runs ETA=$(round(eta/60, digits=1))min")
                catch e
                    println("  FAILED [$wname] dg=$delta_G lie=$lie_level rep=$repeat: $(typeof(e)): $e")
                end
            end
        end
    end
end
end # let done_runs

# ---------- Summary ----------
println("\n============================================================")
println("Summary: subset=$(SUBSET_IDS)  liar_idx=$LIAR_IDX (building $(SUBSET_IDS[LIAR_IDX]))  num_repeats=$num_repeats")
println("============================================================")
open(OUT_SUM, "w") do io
    println(io, "sweep_lying_agent summary")
    println(io, "subset_ids = $(SUBSET_IDS)")
    println(io, "liar_idx = $LIAR_IDX (building $(SUBSET_IDS[LIAR_IDX]))")
    println(io, "lie_levels = $(LIE_LEVELS)")
    println(io, "lie_bound = $LIE_BOUND")
    println(io, "num_repeats = $num_repeats")
    println(io, "delta_G_values = $(delta_G_values)")
    println(io, "windows = $([w[1] for w in WINDOWS])")
    println(io, "dec_avg = $dec_avg")
    println(io)
    for (wname, _, _) in WINDOWS
        sub = filter(:window => ==(wname), data)
        println(io, "=== Window $wname ===")
        for ll in LIE_LEVELS
            sub_ll = filter(:lie_level => ==(ll), sub)
            gdf = combine(groupby(sub_ll, :delta_G),
                         :honest_avg_cost => mean => :mean_honest,
                         :honest_avg_cost => std  => :std_honest,
                         :liar_cost       => mean => :mean_liar,
                         :total_avg_cost  => mean => :mean_total,
                         :clip_fraction   => mean => :mean_clip,
                         nrow => :n_runs)
            println(io, "\n  lie_level = $ll")
            for r in eachrow(gdf)
                println(io, "    dg=$(r.delta_G)  honest=$(round(r.mean_honest, digits=2))±$(round(r.std_honest, digits=2))  liar=$(round(r.mean_liar, digits=2))  total=$(round(r.mean_total, digits=2))  clip=$(round(r.mean_clip, digits=3))  n=$(r.n_runs)")
            end
        end
        # Degradation table: honest cost relative to truthful
        println(io, "\n  Degradation (honest_avg_cost - truthful_honest_avg_cost):")
        truth = combine(groupby(filter(:lie_level => ==("truthful"), sub), :delta_G),
                        :honest_avg_cost => mean => :truthful_honest)
        for ll in ["bounded", "fabrication"]
            ll_df = combine(groupby(filter(:lie_level => ==(ll), sub), :delta_G),
                            :honest_avg_cost => mean => :lie_honest)
            deg = innerjoin(truth, ll_df, on=:delta_G)
            deg.degradation = deg.lie_honest .- deg.truthful_honest
            println(io, "    $ll:")
            for r in eachrow(deg)
                println(io, "      dg=$(r.delta_G)  degradation=$(round(r.degradation, digits=2))  ($(round(100*r.degradation/abs(r.truthful_honest), digits=2))%)")
            end
        end
    end
end
println("Summary written to $OUT_SUM")
println("Done.")
