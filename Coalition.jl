include("MPC_optimiser.jl")
using Combinatorics
using StatsBase
using Base.Threads

# Return the building-id list for an agent (Int singleton or Vector coalition)
agent_ids(a::Int) = [a]
agent_ids(a::Vector) = a


function find_opt_coal(buildings::Vector{Building}, max_coal_size::Int)
    possible_coals = collect(partitions(buildings))
    coal_vals = Dict()
    for coal in possible_coals
	    valid_coal = true
	    for elem in coal
		    if elem.size[1] > max_coal_size
			    valid_coal = false
			    break
		    end
	    end
	    if valid_coal
            obj_val=0
            for ind_coal in coal
                coal_res = optimise(opt,ind_coal)
                obj_val += sum(value(coal_res[2][5]))
            end
            coal_vals[coal] = obj_val
	    end
    end
    sorted_coal_vals = sort!(collect(coal_vals), by=last)
    return sorted_coal_vals[1][1], sorted_coal_vals[1][2], length(possible_coals)
end

function bottom_up_full_info(buildings::Vector{Building}, max_coal_size::Int)
    agents = Vector(1:length(buildings))
    done = false
    outs = [optimise(opt, buildings[agent]) for agent in agents]
    num_iters = 1

    coal_vars = Dict()
    coal_vals = Dict()
    for (out, agent) in zip(outs, agents)
        coal_vars[agent] = out[1]
        coal_vals[[agent]] = sum(value(out[2][5]))
    end
    while !done
        done = true
        #for agent in agents
        #    println(optimise(opt, buildings[agent])[1])
        #end
        outs = [optimise(opt, buildings[agent]) for agent in agents]
        res = Dict([agent => out[1] for (out, agent) in zip(outs, agents)])
        vars = [out[2] for out in outs]

        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        poss_coal_vars = Dict()
        for c in poss_coals
            poss_coal_vec = Vector()
            append!(poss_coal_vec, c[1], c[2])
            coal_res = optimise(opt, buildings[poss_coal_vec])
            obj_val = sum(value(coal_res[2][5]))
            poss_coal_vars[c] = coal_res[1]
            if c[1] isa Int64
                c_1 = [c[1]]
            else
                c_1 = c[1]
            end
            if c[2] isa Int64
                c_2 = [c[2]]
            else
                c_2 = c[2]
            end
            poss_coal_vals[c] = -((coal_vals[c_1])+(coal_vals[c_2]))+ obj_val
            # poss_coal_vals[c] = -(objective_value(res[c[1]])+objective_value( res[c[2]]))+ sum(objective_value(optimise(opt, buildings[poss_coal_vec])[1])) #added res c[1] c[2] - joint to better discriminate
        end
        sorted_coal_vals = sort!(collect(poss_coal_vals), by=last)
        new_agents = Vector()
        coaled_agents = Vector()
        for elem in sorted_coal_vals
            if !(elem[1][1] in coaled_agents) && !(elem[1][2] in coaled_agents)
                new_coal = Vector()
                append!(new_coal, elem[1][1], elem[1][2])
                if length(new_coal) <= max_coal_size
                    push!(new_agents, new_coal)
                    push!(coaled_agents, elem[1][1], elem[1][2])
                    coal_vals[new_coal] = poss_coal_vals[elem[1]]
                    coal_vars[new_coal] = poss_coal_vars[elem[1]]
                    done = false
                end
            end
        end
        for agent in agents
            if !(agent in coaled_agents)
                push!(new_agents, agent)
            end
        end

        agents = new_agents
        #println(sorted_coal_vals)
        #update cons_vec
    end

    return agents, [coal_vars[agent] for agent in agents], num_iters

end

function privacy_focussed_coals(buildings::Vector{Building}, max_coal_size::Int)
    agents = Vector(1:length(buildings))
    done = false
    energy_diff = opt.energy_cost-opt.energy_sale
    while !done
        done = true
        #for agent in agents
        #    println(optimise(opt, buildings[agent])[1])
        #end
        outs = [optimise(opt, buildings[agent]) for agent in agents]
        res = [out[1] for out in outs]
        vars = [out[2] for out in outs]

        cons_vec = Dict([agent=> vec(sum(value(var[2]-var[3]),dims=2)) for (agent, var) in zip(agents,vars)])

        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        for c in poss_coals
            compatible_slots = cons_vec[c[1]].*cons_vec[c[2]] .< zeros(length(cons_vec[c[1]]))
            if any(compatible_slots)
                poss_coal_vals[c] = energy_diff*(min(abs.(compatible_slots.*cons_vec[c[1]]),abs.(compatible_slots.*cons_vec[c[2]]))) # should be dotted with price vector
                #poss_coal_vals[c] = norm(cons_vec[c[1]]+cons_vec[c[2]])
            end
        end
        # sorted_coal_vals = sort!(collect(poss_coal_vals), by=last)
        new_agents = Vector()
        coaled_agents = Vector()
        done = false
        delta_u = 1 # change this!
        println("Needs edit")
        epsilon = 1e-1
        weights = [Float64(exp(epsilon*i[2]/(2*delta_u))) for i in sorted_coal_vals]
        poss_coals_for_samples = [elem[1] for elem in sorted_coal_vals]
        while not done
            elem = sample(poss_coal_vals, Weights(weights))
            if !(elem[1] in coaled_agents) && !(elem[2] in coaled_agents)
                new_coal = Vector()
                append!(new_coal, elem[1], elem[2])
                if length(new_coal) <= max_coal_size
                    push!(new_agents, new_coal)
                    push!(coaled_agents, elem[1], elem[2])
                end
            end
            ind = findfirst(==(elem), poss_coal_vals)
            poss_coals_for_samples = vcat(poss_coals_for_samples[1:ind-1], poss_coals_for_samples[ind+1:length(poss_coals_for_samples)])
            weights = vcat(weights[1:ind-1], weights[ind+1:length(weights)])
            if length(poss_coals_for_samples) == 0
                done = true
            end
        end
        # for elem in sorted_coal_vals
        #     if !(elem[1][1] in coaled_agents) && !(elem[1][2] in coaled_agents)
        #         new_coal = Vector()
        #         append!(new_coal, elem[1][1], elem[1][2])
        #         if length(new_coal) <= max_coal_size
        #             push!(new_agents, new_coal)
        #             push!(coaled_agents, elem[1][1], elem[1][2])
        #             done = false
        #         end
        #     end
        # end
        for agent in agents
            if !(agent in coaled_agents)
                push!(new_agents, agent)
            end
        end

        agents = new_agents
        #println(sorted_coal_vals)
        #update cons_vec
    end
    #println(cons_vec)
    #println("cons ", value(vars[2][2]))
    # #println("sell ", vars[1][3])
    # outs = [optimise(opt, buildings[agent]) for agent in agents]
    # res = [sum(objective_value(out[1])) for out in outs]
    # println(agents)
    outs = [optimise(opt, buildings[agent]) for agent in agents]
    res = [out[1] for out in outs]
    vars = [sum(value(out[2][5])) for out in outs]
    return agents, vars, 1
end

function find_opt_coal(buildings::Vector{MPC_Building}, max_coal_size::Int,k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
    possible_coals = collect(partitions(buildings))
    coal_vals = Dict()
    num_iters = 0
    for coal in possible_coals
	    valid_coal = true
	    for elem in coal
		    if elem.size[1] > max_coal_size
			    valid_coal = false
			    break
		    end
	    end
	    if valid_coal
            obj_val = 0
            for ind_coal in coal
                coal_res = single_optimise_ADMM(opt,ind_coal,k,num_look_ahead,receding_horizon)
                num_iters += coal_res[2]
                obj_val += sum(value(coal_res[1][5]))
            end
            coal_vals[coal] = obj_val
	    end
    end
    sorted_coal_vals = sort!(collect(coal_vals), by=last)
    outs = [single_optimise_ADMM(opt, agent, k)[1] for agent in sorted_coal_vals[1][1]] #can do this in the loop, so don't count these iters
    return sorted_coal_vals[1][1], outs, num_iters#, sorted_coal_vals[1][2]
end

function bottom_up_full_info(buildings::Vector{MPC_Building}, max_coal_size::Int, k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
    agents = Vector(1:length(buildings))
    done = false
    outs = [single_optimise_ADMM(opt, buildings[agent], k,num_look_ahead,receding_horizon) for agent in agents]
    num_iters = sum([out[2] for out in outs])

    coal_vars = Dict()
    coal_vals = Dict()
    for (out, agent) in zip(outs, agents)
        coal_vars[agent] = out[1]
        coal_vals[[agent]] = sum(value(out[1][5]))
    end
    while !done
        done = true
        #for agent in agents
        #    println(optimise(opt, buildings[agent])[1])
        #end
        

        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        poss_coal_vars = Dict()
        for c in poss_coals
            poss_coal_vec = Vector()
            append!(poss_coal_vec, c[1], c[2])
            coal_res = single_optimise_ADMM(opt, buildings[poss_coal_vec], k,num_look_ahead,receding_horizon)
            num_iters += coal_res[2]
            obj_val = sum(value(coal_res[1][5]))
            poss_coal_vars[c] = coal_res[1]
            if c[1] isa Int64
                c_1 = [c[1]]
            else
                c_1 = c[1]
            end
            if c[2] isa Int64
                c_2 = [c[2]]
            else
                c_2 = c[2]
            end
            poss_coal_vals[c] = -((coal_vals[c_1])+(coal_vals[c_2]))+ obj_val #added poss_coal_vals c[1] c[2] - joint to better discriminate
            # poss_coal_vals[c] = -(objective_value(res[c[1]])+objective_value( res[c[2]]))+ sum(obj_val) #added res c[1] c[2] - joint to better discriminate
        end
        sorted_coal_vals = sort!(collect(poss_coal_vals), by=last)
        new_agents = Vector()
        coaled_agents = Vector()
        for elem in sorted_coal_vals
            if !(elem[1][1] in coaled_agents) && !(elem[1][2] in coaled_agents)
                new_coal = Vector()
                append!(new_coal, elem[1][1], elem[1][2])
                if length(new_coal) <= max_coal_size
                    push!(new_agents, new_coal)
                    push!(coaled_agents, elem[1][1], elem[1][2])
                    coal_vals[new_coal] = poss_coal_vals[elem[1]]
                    coal_vars[new_coal] = poss_coal_vars[elem[1]]
                    done = false
                end
            end
        end
        for agent in agents
            if !(agent in coaled_agents)
                push!(new_agents, agent)
            end
        end

        agents = new_agents
        #println(sorted_coal_vals)
        #update cons_vec
    end
    #println(cons_vec)
    #println("cons ", value(vars[2][2]))
    #println("sell ", vars[1][3])
    # outs = [single_optimise(opt, buildings[agent], k) for agent in agents]
    # res = [sum(objective_value(out[1])) for out in outs]
    return agents, [coal_vars[agent] for agent in agents], num_iters
end

function privacy_focussed_coals_with_delta(buildings::Vector{MPC_Building}, max_coal_size::Int, k::Int,num_look_ahead::Int,receding_horizon::Bool=false, delta_G::Float64=100.0)
    agents = Vector(1:length(buildings))
    done = false
    energy_diff = opt.energy_cost-opt.energy_sale
    # Compute correct delta_u bound: max timestep difference (buy price - sell price) * delta_G
    max_price_diff = maximum(abs.(opt.energy_cost - opt.energy_sale))
    # Add minimum threshold to prevent numerical issues with very small delta_G
    delta_u = max(delta_G * max_price_diff, 1e-10)
    num_iters=0
    vars = 0
    dec_single_vals = 0
    # P1.2: Cache singleton results: building_id => (vars, cons_vec)
    singleton_cache = Dict{Int, Tuple{Any, Vector{Float64}}}()
    cache_lock = SpinLock()
    first_round = true
    # dec_outs_single = [single_optimise_ADMM(opt, buildings[agent], k,num_look_ahead,receding_horizon)[1] for agent in agents]
    # dec_single_vals = [value(out[2][1])*energy_cost_k(opt,k,1)-value(out[3][1])*energy_sale_k(opt,k,1) for out in dec_outs_single]
    while !done
        done = true
        # P1.2 + P1.5: Cache singleton solves across rounds; parallelise across agents.
        outs = Vector{Any}(undef, length(agents))
        @threads for i in 1:length(agents)
            agent = agents[i]
            if agent isa Int
                # Singleton: check cache (with lock for read), solve + cache if missing
                local cv
                cached = false
                lock(cache_lock) do
                    cached = haskey(singleton_cache, agent)
                    if cached
                        outs[i] = (singleton_cache[agent][1], 0)
                    end
                end
                if !cached
                    res = single_optimise_ADMM(opt, buildings[agent], k, num_look_ahead, receding_horizon)
                    cv = vec(sum(value(res[1][2]-res[1][3]), dims=2))
                    lock(cache_lock) do
                        singleton_cache[agent] = (res[1], cv)
                    end
                    outs[i] = (res[1], res[2])
                end
            else
                res = single_optimise_ADMM(opt, buildings[[agent...]], k, num_look_ahead, receding_horizon)
                outs[i] = (res[1], res[2])
            end
        end
        vars = [out[1] for out in outs]
        if first_round
            # dec_single_vals[i] = standalone first-step cost for building i
            dec_single_vals = [value(outs[i][1][2][1])*energy_cost_k(opt,k,1) - value(outs[i][1][3][1])*energy_sale_k(opt,k,1) for i in 1:length(buildings)]
            first_round = false
        end
        num_iters += sum([out[2] for out in outs])
        #res = [out[1] for out in outs]

        cons_vec = Dict{Any, Vector{Float64}}()
        for (agent, var) in zip(agents, vars)
            if agent isa Int && haskey(singleton_cache, agent)
                cons_vec[agent] = singleton_cache[agent][2]
            else
                cons_vec[agent] = vec(sum(value(var[2]-var[3]), dims=2))
            end
        end

        # P1.4: Vectorise the O(n^2) pair compatibility check.
        # Build cons_mat (horizon × n_agent) and compute all pair values with matrix ops.
        n_agents = length(agents)
        horizon = min(length(buildings[1].act_cons)-k+1, num_look_ahead)
        cons_mat = reduce(hcat, [cons_vec[a][1:horizon] for a in agents])  # horizon × n_agent
        abs_mat = abs.(cons_mat)  # horizon × n_agent
        # sign_compat[t,i,j] = cons_mat[t,i]*cons_mat[t,j] < 0  (opposite signs => can trade)
        # min_pair[t,i,j]    = min(abs_mat[t,i], abs_mat[t,j])
        # Use 3D broadcasting: abs_mat (h×n×1) vs abs_mat' reshaped (1×n×h)? Easier: explicit loops over t (horizon is small, ~2-8).
        diff_vec = energy_diff[1:horizon]
        pair_val_mat = zeros(n_agents, n_agents)
        any_compat = falses(n_agents, n_agents)
        for t in 1:horizon
            row = cons_mat[t, :]            # n_agent
            opp = (row .* row') .< 0        # n_agent × n_agent: signs differ
            m = min.(abs.(row), abs.(row')) # n_agent × n_agent: pairwise min of magnitudes
            pair_val_mat .+= diff_vec[t] .* (opp .* m)
            any_compat .|= opp
        end
        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        for c in poss_coals
            i = findfirst(==(c[1]), agents)
            j = findfirst(==(c[2]), agents)
            if any_compat[i, j]
                poss_coal_vals[c] = pair_val_mat[i, j]
            end
        end
        # sorted_coal_vals = sort!(collect(poss_coal_vals), by=last)
        #
        new_agents = Vector()
        # P2.1: Track whether any merge was actually accepted this round; if not, exit.
        merge_happened = false
        if length(poss_coal_vals) == 0
            new_agents = agents
            break
        end
        epsilon = 1e-1
        # Add numerical stability to prevent overflow in exp()
        coal_weights = Dict{Any, Float64}()
        for i in keys(poss_coal_vals)
            # Safe weight calculation with bounds checking
            weight_val = epsilon*poss_coal_vals[i]/(2*delta_u)
            # Clip extreme values to prevent overflow
            capped_weight = min(weight_val, 50.0)  # Prevent overflow in exp()
            coal_weights[i] = exp(capped_weight)
        end
        ks = collect(keys(coal_weights))
        weights = [coal_weights[k] for k in ks]
        # Ensure weights are finite and valid
        weights = [isfinite(w) ? w : 1.0 for w in weights]
        n_pool = length(ks)
        # Phase A: instead of a coaled_agents membership check on every draw,
        # when a merge is accepted we eagerly remove ALL pairs overlapping
        # with the new coalition from the pool. This shrinks the pool faster
        # (so each subsequent sample(1:n_pool, Weights(...)) builds a smaller
        # alias table) and eliminates the O(|coaled|) membership check.
        while n_pool > 0
            try
                ind = sample(1:n_pool, Weights(weights[1:n_pool]))
                elem = ks[ind]
                # Remove the drawn pair (swap-with-last, O(1))
                weights[ind] = weights[n_pool]
                ks[ind] = ks[n_pool]
                n_pool -= 1
                # Form the candidate coalition as a flat id list
                new_coal = vcat(agent_ids(elem[1]), agent_ids(elem[2]))
                if length(new_coal) <= max_coal_size
                    push!(new_agents, new_coal)
                    merge_happened = true
                    # Eagerly remove every pair in the pool that overlaps with new_coal.
                    # Each pair is removed at most once per round -> O(n^2) total.
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
            catch e
                # In case of weight problems, fall back to uniform sampling
                if length(ks) > 0
                    # If we're in this catch block, there might be issues with weights
                    # Fall back to choosing a random pair
                    ind = rand(1:min(n_pool, length(ks)))
                    elem = ks[ind]
                    # Remove the drawn pair (swap-with-last, O(1))
                    weights[ind] = weights[n_pool]
                    ks[ind] = ks[n_pool]
                    n_pool -= 1
                    # Form the candidate coalition as a flat id list
                    new_coal = vcat(agent_ids(elem[1]), agent_ids(elem[2]))
                    if length(new_coal) <= max_coal_size
                        push!(new_agents, new_coal)
                        merge_happened = true
                    end
                else
                    break
                end
            end
        end
        # Phase A.3: derive coaled_ids from the accepted coalitions in new_agents
        coaled_ids = Set{Int}()
        for na in new_agents
            union!(coaled_ids, agent_ids(na))
        end
        for agent in agents
            if !any(id in coaled_ids for id in agent_ids(agent))
                push!(new_agents, agent)
            end
        end

        agents = new_agents
        # P2.1: Exit when no merge was accepted this round (instead of running until poss_coal_vals is empty)
        done = !merge_happened
        #println(sorted_coal_vals)
        #update cons_vec
    end
    num_problems = 0
    new_agents = Vector()
    added = false
    # P1.3: Map each current agent to its var, so we can reuse results for kept coalitions
    vars_by_agent = Dict{Any, Any}(zip(agents, vars))
    for (agent, var) in zip(agents,vars)
        added = false
        if length(agent) > 1
            dec_val = sum(sum(dec_single_vals[i] for i in agent))
            coal_single_val = sum(value(var[2][1,:]).*energy_cost_k(opt,k,1)-value(var[3][1,:]).*energy_sale_k(opt,k,1))
            if dec_val >= coal_single_val
                push!(new_agents, agent)
                added = true
            else
                for i in agent
                    push!(new_agents, i)
                end
            end
        else
            push!(new_agents,agent)
        end
    end
    if added
        agents = new_agents
    end
    # P1.3: Reuse cached/reused vars where possible; only solve genuinely new singletons (from splits)
    vars = Vector{Any}(undef, length(agents))
    for (i, agent) in enumerate(agents)
        if haskey(vars_by_agent, agent)
            # Kept coalition or unchanged singleton: reuse existing result
            vars[i] = vars_by_agent[agent]
        elseif agent isa Int && haskey(singleton_cache, agent)
            # Singleton split out of a coalition but we already solved it standalone earlier
            vars[i] = singleton_cache[agent][1]
        else
            # Genuinely new singleton from a split — solve now and cache
            res = single_optimise_ADMM(opt, buildings[agent], k, num_look_ahead, receding_horizon)
            num_iters += res[2]
            cv = vec(sum(value(res[1][2]-res[1][3]), dims=2))
            singleton_cache[agent] = (res[1], cv)
            vars[i] = res[1]
        end
    end
    return agents, vars, num_iters
end

function privacy_focussed_coals(buildings::Vector{MPC_Building}, max_coal_size::Int, k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
    # Wrapper for backward compatibility with existing code
    return privacy_focussed_coals_with_delta(buildings, max_coal_size, k, num_ahead, receding_horizon, 100.0)
end