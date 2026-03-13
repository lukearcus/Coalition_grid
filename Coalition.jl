include("MPC_optimiser.jl")

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
		    coal_vals[coal] = sum([objective_value(optimise(opt,ind_coal)[1]) for ind_coal in coal])
	    end
    end
    sorted_coal_vals = sort!(collect(coal_vals), by=last)
    return sorted_coal_vals[1][1], sorted_coal_vals[1][2], length(possible_coals)
end

function bottom_up_full_info(buildings::Vector{Building}, max_coal_size::Int)
    agents = Vector(1:length(buildings))
    done = false
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
        for c in poss_coals
            poss_coal_vec = Vector()
            append!(poss_coal_vec, c[1], c[2])
            poss_coal_vals[c] = -(objective_value(res[c[1]])+objective_value( res[c[2]]))+ sum(objective_value(optimise(opt, buildings[poss_coal_vec])[1])) #added res c[1] c[2] - joint to better discriminate
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
    outs = [optimise(opt, buildings[agent]) for agent in agents]
    res = [sum(objective_value(out[1])) for out in outs]
    return agents, sum(res), num_iters
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
    outs = [optimise(opt, buildings[agent]) for agent in agents]
    res = [sum(objective_value(out[1])) for out in outs]
    return agents, sum(res)
end

function find_opt_coal(buildings::Vector{MPC_Building}, max_coal_size::Int,k::Int)
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
            coal_res = single_optimise_ADMM(opt,ind_coal,k)
            num_iters += coal_res[2]
            obj_val = sum(coal_res[1][5])
		    coal_vals[coal] = sum([obj_val for ind_coal in coal])
	    end
    end
    sorted_coal_vals = sort!(collect(coal_vals), by=last)
    outs = [single_optimise_ADMM(opt, buildings[agent], k) for agent in sorted_coal_vals[1][1]] #can do this in the loop, so don't count these iters
    return sorted_coal_vals[1][1], outs, num_iters#, sorted_coal_vals[1][2]
end

function bottom_up_full_info(buildings::Vector{MPC_Building}, max_coal_size::Int, k::Int)
    agents = Vector(1:length(buildings))
    done = false
    outs = [single_optimise_ADMM(opt, buildings[agent], k) for agent in agents]
    num_iters = sum([out[2] for out in outs])

    coal_vars = Dict()
    poss_coal_vals = Dict()
    for (out, agent) in zip(outs, agents)
        coal_vars[[agent]] = out[1]
        poss_coal_vals[[agent]] = sum(value(out[1][5]))
    end

    while !done
        done = true
        #for agent in agents
        #    println(optimise(opt, buildings[agent])[1])
        #end
        

        poss_coals = collect(combinations(agents,2))
        for c in poss_coals
            poss_coal_vec = Vector()
            append!(poss_coal_vec, c[1], c[2])
            coal_res = single_optimise_ADMM(opt, buildings[poss_coal_vec], k)
            num_iters += coal_res[2]
            obj_val = sum(value(coal_res[1][5]))
            coal_vars[c] = coal_res[1]
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
            poss_coal_vals[c] = -((poss_coal_vals[c_1])+( poss_coal_vals[c_2]))+ obj_val #added poss_coal_vals c[1] c[2] - joint to better discriminate
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

function privacy_focussed_coals(buildings::Vector{MPC_Building}, max_coal_size::Int, k::Int)
    agents = Vector(1:length(buildings))
    done = false
    energy_diff = opt.energy_cost-opt.energy_sale
    num_iters=0
    vars = 0
    while !done
        done = true
        #for agent in agents
        #    println(optimise(opt, buildings[agent])[1])
        #end
        outs = [single_optimise_ADMM(opt, buildings[agent], k) for agent in agents]
        num_iters += sum([out[2] for out in outs])
        #res = [out[1] for out in outs]
        vars = [out[1] for out in outs]

        cons_vec = Dict([agent=> vec(sum(value(var[2]-var[3]),dims=2)) for (agent, var) in zip(agents,vars)])

        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        for c in poss_coals
            num_steps = length(buildings[1].act_cons)-k+1
            compatible_slots = (cons_vec[c[1]].*cons_vec[c[2]] .< zeros(length(cons_vec[c[1]])))[1:num_steps]
            if any(compatible_slots)
                poss_coal_vals[c] = energy_diff[1:num_steps]'*(min(abs.(compatible_slots.*cons_vec[c[1]]),abs.(compatible_slots.*cons_vec[c[2]]))) # should be dotted with price vector
                #poss_coal_vals[c] = norm(cons_vec[c[1]]+cons_vec[c[2]])
            end
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
    # outs = [optimise(opt, buildings[agent]) for agent in agents]
    # res = [sum(objective_value(out[1])) for out in outs]
    return agents, vars, num_iters
end