# Regression test for coalition formation.
#
# Three test cases exercise both paradigms:
#   Case 1 (non-DP, singleton path): 8 buildings, k=1, horizon=2, 4 steps.
#     No compatible pairs -> no merges -> deterministic. Asserts exact
#     coalition match and tight cost tolerance (1e-3 relative).
#   Case 2 (non-DP, merge path): 8 buildings, k=40, horizon=8, 96 steps.
#     Compatible pairs -> greedy merges + split-check. Deterministic, so
#     asserts exact coalition match and tight cost tolerance (1e-3 relative).
#   Case 3 (DP, merge path): 8 buildings, k=40, horizon=8, 96 steps, delta_G=100.
#     Exponential-mechanism sampling -> coalition structure is RNG-dependent.
#     Asserts validity (every building in exactly one coalition, no coalition
#     exceeds max_coal_size) and cost within a loose tolerance (20% relative)
#     of the baseline, since the per-seed coalition structure varies.
#
# Usage:
#   julia --threads auto test_regression.jl              # run all cases
#   julia --threads auto test_regression.jl --baseline  # record baselines
#   julia --threads auto test_regression.jl --case 1     # run only case 1
include("Buildings.jl")
include("MPC_optimiser.jl")
include("Coalition.jl")
include("load_EMS_data.jl")

using Random
using Printf

# agent_ids mirrored here so the test is self-contained for validity checks
_aid(a::Int) = [a]
_aid(a::Vector) = a

# ---- Test case configurations ----
struct CaseConfig
    name::String
    num_builds::Int
    max_coal::Int
    k::Int
    horizon::Int
    num_steps::Int
    seed::Int
    method::Symbol          # :nondp -> privacy_focussed_coals; :dp -> privacy_focussed_coals_with_delta
    delta_G::Float64        # privacy budget (only used by :dp)
    exact_match::Bool       # assert exact coalition match (deterministic cases)
    cost_tol_rel::Float64   # relative cost tolerance
    baseline_file::String
end

const CASES = [
    CaseConfig("nondp_singleton", 8, 6, 1, 2, 4, 42, :nondp, 0.0, true, 1e-3, "results/baseline_regression.txt"),
    CaseConfig("nondp_merges",   8, 6, 40, 8, 96, 42, :nondp, 0.0, true, 1e-3, "results/baseline_regression_k40.txt"),
    CaseConfig("dp_merges",      8, 6, 40, 8, 96, 42, :dp, 100.0, false, 0.20, "results/baseline_regression_dp.txt"),
]

# ---- Helpers ----
function coal_key(coal)
    out = Vector{Vector{Int}}()
    for agent in coal
        push!(out, sort(_aid(agent)))
    end
    return sort(out)
end

function coal_to_str(c)
    return join(["[" * join(string.(v), ",") * "]" for v in c], " ")
end

# Validity: every building id in 1:num_builds appears in exactly one coalition,
# and no coalition exceeds max_coal.
function is_valid(coal, num_builds, max_coal)
    seen = Set{Int}()
    for agent in coal
        ids = _aid(agent)
        if length(ids) > max_coal
            return false, "coalition $ids exceeds max_coal=$max_coal"
        end
        for id in ids
            if id in seen
                return false, "building $id in multiple coalitions"
            end
            if !(1 <= id <= num_builds)
                return false, "building id $id out of range"
            end
            push!(seen, id)
        end
    end
    if length(seen) != num_builds
        return false, "$(num_builds - length(seen)) buildings missing from coalitions"
    end
    return true, ""
end

function run_case(cfg::CaseConfig)
    buildings, energy_cost, energy_sale, timestamps = MPC_load_from_CSV(cfg.num_builds, cfg.num_steps)
    global opt = MPC_optimiser(energy_cost', energy_sale')
    Random.seed!(cfg.seed)
    if cfg.method == :nondp
        coal, vars, num_iters = privacy_focussed_coals(buildings, cfg.max_coal, cfg.k, cfg.horizon)
    else
        coal, vars, num_iters = privacy_focussed_coals_with_delta(buildings, cfg.max_coal, cfg.k, cfg.horizon, false, cfg.delta_G)
    end
    total_cost = 0.0
    for (agent, var) in zip(coal, vars)
        total_cost += sum(value(var[5]))
    end
    return Dict(
        "coalition_str" => coal_to_str(coal_key(coal)),
        "total_cost"    => total_cost,
        "num_iters"     => num_iters,
        "coal"          => coal,
    )
end

function write_baseline(cfg::CaseConfig, res)
    open(cfg.baseline_file, "w") do io
        println(io, "coalition_str=", res["coalition_str"])
        println(io, "total_cost=", res["total_cost"])
        println(io, "num_iters=", res["num_iters"])
    end
end

function read_baseline(cfg::CaseConfig)
    baseline = Dict{String,Any}()
    for line in eachline(cfg.baseline_file)
        k, v = split(line, "="; limit=2)
        if k == "total_cost"
            baseline[k] = parse(Float64, v)
        elseif k == "num_iters"
            baseline[k] = parse(Int, v)
        else
            baseline[k] = v
        end
    end
    return baseline
end

function check_case(cfg::CaseConfig, res, baseline)
    pass = true
    # 1. Validity (always checked)
    ok, msg = is_valid(res["coal"], cfg.num_builds, cfg.max_coal)
    if ok
        println("  [ok]   validity (all buildings in exactly one coalition, sizes <= max_coal)")
    else
        println("  [FAIL] validity: $msg")
        pass = false
    end
    # 2. Coalition match (only for deterministic cases)
    if cfg.exact_match
        if res["coalition_str"] == baseline["coalition_str"]
            println("  [ok]   coalition structure matches baseline")
        else
            @printf "  [FAIL] coalition structure changed\n    baseline: %s\n    new:      %s\n" baseline["coalition_str"] res["coalition_str"]
            pass = false
        end
    else
        println("  [info] stochastic case — skipping exact coalition match (RNG-dependent)")
    end
    # 3. Cost tolerance
    cost_delta = abs(res["total_cost"] - baseline["total_cost"])
    cost_threshold = cfg.cost_tol_rel * max(1.0, abs(baseline["total_cost"]))
    if cost_delta <= cost_threshold
        @printf "  [ok]   total_cost within tolerance (Δ=%.6e <= %.2e)\n" cost_delta cost_threshold
    else
        @printf "  [FAIL] total_cost outside tolerance (Δ=%.6e > %.2e)\n" cost_delta cost_threshold
        pass = false
    end
    return pass
end

function main()
    only_case = nothing
    for i in 1:length(ARGS)-1
        if ARGS[i] == "--case"
            only_case = parse(Int, ARGS[i+1])
        end
    end
    is_baseline = "--baseline" in ARGS
    mkpath("results")
    all_pass = true
    for (idx, cfg) in enumerate(CASES)
        if only_case !== nothing && idx != only_case
            continue
        end
        @printf "=== Case %d (%s): %d builds, k=%d, horizon=%d ===\n" idx cfg.name cfg.num_builds cfg.k cfg.horizon
        if is_baseline
            res = run_case(cfg)
            write_baseline(cfg, res)
            @printf "  Wrote baseline: %s\n" cfg.baseline_file
            @printf "    coalition: %s\n" res["coalition_str"]
            @printf "    total_cost: %.6f\n" res["total_cost"]
            @printf "    num_iters: %d\n" res["num_iters"]
        else
            if !isfile(cfg.baseline_file)
                @printf "  [SKIP] no baseline at %s — run with --baseline first\n" cfg.baseline_file
                all_pass = false
                continue
            end
            baseline = read_baseline(cfg)
            res = run_case(cfg)
            @printf "  baseline: coalition=%s cost=%.6f iters=%d\n" baseline["coalition_str"] baseline["total_cost"] baseline["num_iters"]
            @printf "  new:      coalition=%s cost=%.6f iters=%d\n" res["coalition_str"] res["total_cost"] res["num_iters"]
            if !check_case(cfg, res, baseline)
                all_pass = false
            end
        end
        println()
    end
    @printf "Overall: %s\n" (all_pass ? "PASS" : "FAIL")
    exit(all_pass ? 0 : 1)
end

main()
