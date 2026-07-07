# Unit test: verify the Phase A (overlap-removal) sampling is a valid and
# distributionally-equivalent replacement for the pre-refactor
# (membership-check) version.
#
# Checks:
#   1. VALIDITY: every agent appears in exactly one coalition in the output,
#      and no coalition exceeds max_coal_size.
#   2. DISTRIBUTIONAL EQUIVALENCE: over many random seeds, the empirical
#      distribution of coalition structures matches between old and new.
#      (Per-seed exact match is NOT expected: the two implementations consume
#      different numbers of RNG draws, so the same seed produces different
#      coalition structures. But the statistical distribution is identical
#      because the conditional probability of each accepted coalition, given
#      the current accepted set, is the same in both versions.)
#
# Run: julia --project=... test_sampling.jl

include("sampling.jl")
using Random
using Printf
using StatsBase

function make_poss_coal_vals(pairs_vals::Vector{Tuple{Tuple{Int,Int}, Float64}})
    d = Dict{Tuple{Int,Int}, Float64}()
    for (p, v) in pairs_vals
        d[p] = v
    end
    return d
end

const SEEDS = [42, 1, 7, 12345, 999, 2024, 31337, 0]

function scenario1()
    agents = collect(1:6)
    pairs = [(i, j) for i in 1:6 for j in i+1:6]
    vals = Dict(p => 1.0 for p in pairs)
    return (agents, vals, 6, "6 agents, all pairs compatible, uniform")
end

function scenario2()
    agents = collect(1:6)
    pairs = [(i, j) for i in 1:6 for j in i+1:6]
    vals = Dict(p => (p == (1, 2) ? 10.0 : 0.1) for p in pairs)
    return (agents, vals, 6, "6 agents, skewed values (pair (1,2) favoured)")
end

function scenario3()
    agents = collect(1:8)
    compatible_pairs = [(1,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),(1,8),(2,7),(3,6)]
    vals = Dict(p => float(i) for (i, p) in enumerate(compatible_pairs))
    return (agents, vals, 8, "8 agents, sparse compatibility (10 of 28 pairs)")
end

function scenario4()
    agents = collect(1:6)
    pairs = [(i, j) for i in 1:6 for j in i+1:6]
    vals = Dict(p => 1.0 for p in pairs)
    return (agents, vals, 2, "6 agents, max_coal_size=2 (allows pairs, blocks triples)")
end

function scenario5()
    agents = Any[1, [2, 3], 4, [5, 6]]
    pairs = [(agents[1], agents[2]), (agents[1], agents[3]),
             (agents[2], agents[3]), (agents[2], agents[4]),
             (agents[3], agents[4]), (agents[1], agents[4])]
    vals = Dict(p => 1.0 for p in pairs)
    return (agents, vals, 4, "mixed agents (Int + Vector), all compatible")
end

const SCENARIOS = [scenario1, scenario2, scenario3, scenario4, scenario5]

# Validity check: every input agent id appears exactly once in the output,
# and no coalition exceeds max_coal_size.
function is_valid(new_agents, agents, max_coal_size)
    all_ids = Set{Int}()
    for a in agents
        union!(all_ids, agent_ids(a))
    end
    seen = Set{Int}()
    for na in new_agents
        na_ids = agent_ids(na)
        if length(na_ids) > max_coal_size
            return false, "coalition $na_ids exceeds max_coal_size=$max_coal_size"
        end
        for id in na_ids
            if id in seen
                return false, "agent $id appears in multiple coalitions"
            end
            if !(id in all_ids)
                return false, "agent $id not in input"
            end
            push!(seen, id)
        end
    end
    if seen != all_ids
        return false, "missing agents: $(setdiff(all_ids, seen))"
    end
    return true, ""
end

# Convert coalition structure to a canonical string key for distribution comparison
function coal_key(new_agents)
    return join(sort(["[" * join(string.(sort(collect(agent_ids(na)))), ",") * "]" for na in new_agents]), " ")
end

function run_scenario(name, agents, vals, max_coal_size)
    # 1. Validity across all seeds for both versions
    valid_fail = 0
    for seed in SEEDS
        for (label, fn) in [("old", sample_coals_old), ("new", sample_coals_new)]
            out = fn(vals, agents, max_coal_size, MersenneTwister(seed))
            ok, msg = is_valid(out, agents, max_coal_size)
            if !ok
                valid_fail += 1
                @printf "  [VALIDITY FAIL] %s seed=%d %s: %s\n" name seed label msg
            end
        end
    end

    # 2. Distributional equivalence: run N times with random seeds, compare
    #    the empirical distribution of coalition structures via a chi-square-
    #    style check on the most common structures.
    N = 2000
    old_counts = Dict{String, Int}()
    new_counts = Dict{String, Int}()
    for s in 1:N
        seed = s  # distinct seeds
        old = sample_coals_old(vals, agents, max_coal_size, MersenneTwister(seed))
        new = sample_coals_new(vals, agents, max_coal_size, MersenneTwister(seed))
        ko = coal_key(old)
        kn = coal_key(new)
        old_counts[ko] = get(old_counts, ko, 0) + 1
        new_counts[kn] = get(new_counts, kn, 0) + 1
    end
    # Compare distributions: total variation distance between the two empirical distributions
    all_keys = union(keys(old_counts), keys(new_counts))
    tv = 0.0
    for k in all_keys
        p_old = get(old_counts, k, 0) / N
        p_new = get(new_counts, k, 0) / N
        tv += abs(p_old - p_new)
    end
    tv /= 2  # total variation distance is in [0, 1]
    # With N=2000 samples and truly equivalent distributions, TV should be small.
    # Use a tolerance that scales with 1/sqrt(N) (~0.02) plus a margin.
    tv_tol = 0.10
    dist_pass = tv <= tv_tol

    if valid_fail == 0
        @printf "  validity: PASS (%d/%d)\n" (2*length(SEEDS)-valid_fail) (2*length(SEEDS))
    else
        @printf "  validity: FAIL (%d failures)\n" valid_fail
    end
    if dist_pass
        @printf "  distribution: PASS (TV=%.4f <= %.2f)\n" tv tv_tol
    else
        @printf "  distribution: FAIL (TV=%.4f > %.2f)\n" tv tv_tol
        # Show top disagreements
        diffs = sort([(abs(get(old_counts,k,0)-get(new_counts,k,0)), k) for k in all_keys], rev=true)
        for (d, k) in diffs[1:min(5, length(diffs))]
            @printf "    %s: old=%d new=%d\n" k get(old_counts,k,0) get(new_counts,k,0)
        end
    end
    return valid_fail == 0 && dist_pass
end

function main()
    all_pass = true
    for (i, sc) in enumerate(SCENARIOS)
        agents, vals, max_coal_size, desc = sc()
        @printf "Scenario %d (%s):\n" i desc
        ok = run_scenario("scenario $i", agents, vals, max_coal_size)
        if !ok
            all_pass = false
        end
    end
    @printf "\nOverall: %s\n" (all_pass ? "PASS" : "FAIL")
    exit(all_pass ? 0 : 1)
end

main()
