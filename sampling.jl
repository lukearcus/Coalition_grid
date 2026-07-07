# Standalone sampling logic for unit testing the Phase A refactor.
# Mirrors the behaviour of privacy_focussed_coals' sampling loop, both the
# pre-refactor (membership-check) and post-refactor (overlap-removal) versions,
# so they can be compared on a fixed RNG seed.

using StatsBase
using Random

# Return the building-id list for an agent (Int singleton or Vector coalition)
agent_ids(a::Int) = [a]
agent_ids(a::Vector) = a

# ---------------------------------------------------------------------------
# OLD (pre-Phase-A): membership-check version
# Returns new_agents (list of coalitions / leftover singletons)
function sample_coals_old(poss_coal_vals::Dict, agents::Vector, max_coal_size::Int, rng::AbstractRNG)
    new_agents = Vector()
    coaled_agents = Vector()
    delta_u = 1
    epsilon = 1e-1
    coal_weights = Dict([i => exp(epsilon*poss_coal_vals[i]/(2*delta_u)) for i in keys(poss_coal_vals)])
    ks = collect(keys(coal_weights))
    weights = [coal_weights[k] for k in ks]
    n_pool = length(ks)
    while n_pool > 0
        ind = sample(rng, 1:n_pool, Weights(weights[1:n_pool]))
        elem = ks[ind]
        if !(elem[1] in coaled_agents) && !(elem[2] in coaled_agents)
            new_coal = Vector()
            append!(new_coal, elem[1], elem[2])
            if length(new_coal) <= max_coal_size
                push!(new_agents, new_coal)
                push!(coaled_agents, elem[1], elem[2])
            end
        end
        weights[ind] = weights[n_pool]
        ks[ind] = ks[n_pool]
        n_pool -= 1
    end
    for agent in agents
        if !(agent in coaled_agents)
            push!(new_agents, agent)
        end
    end
    return new_agents
end

# ---------------------------------------------------------------------------
# NEW (Phase A): eager overlap-removal version
function sample_coals_new(poss_coal_vals::Dict, agents::Vector, max_coal_size::Int, rng::AbstractRNG)
    new_agents = Vector()
    delta_u = 1
    epsilon = 1e-1
    coal_weights = Dict([i => exp(epsilon*poss_coal_vals[i]/(2*delta_u)) for i in keys(poss_coal_vals)])
    ks = collect(keys(coal_weights))
    weights = [coal_weights[k] for k in ks]
    n_pool = length(ks)
    while n_pool > 0
        ind = sample(rng, 1:n_pool, Weights(weights[1:n_pool]))
        elem = ks[ind]
        weights[ind] = weights[n_pool]
        ks[ind] = ks[n_pool]
        n_pool -= 1
        new_coal = vcat(agent_ids(elem[1]), agent_ids(elem[2]))
        if length(new_coal) <= max_coal_size
            push!(new_agents, new_coal)
            coaled_ids = Set(new_coal)
            i = 1
            while i <= n_pool
                pair_ids = vcat(agent_ids(ks[i][1]), agent_ids(ks[i][2]))
                if !isempty(intersect(pair_ids, coaled_ids))
                    weights[i] = weights[n_pool]
                    ks[i] = ks[n_pool]
                    n_pool -= 1
                else
                    i += 1
                end
            end
        end
    end
    coaled_ids = Set{Int}()
    for na in new_agents
        union!(coaled_ids, agent_ids(na))
    end
    for agent in agents
        if !any(id in coaled_ids for id in agent_ids(agent))
            push!(new_agents, agent)
        end
    end
    return new_agents
end

# Normalise a coalition result to a comparable canonical form:
# sorted list of sorted building-id lists (singletons as 1-vectors)
function canonicalise(coal)
    out = Vector{Vector{Int}}()
    for agent in coal
        push!(out, sort(collect(agent_ids(agent))))
    end
    return sort(out)
end
