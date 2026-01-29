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
    return sorted_coal_vals[1][1], sorted_coal_vals[1][2]
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
        res = [out[1] for out in outs]
        vars = [out[2] for out in outs]

        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        for c in poss_coals
            poss_coal_vec = Vector()
            append!(poss_coal_vec, c[1], c[2])
            poss_coal_vals[c] = sum(objective_value(optimise(opt, buildings[poss_coal_vec])[1]))
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

function privacy_focussed_coals(buildings::Vector{Building}, max_coal_size::Int)
    agents = Vector(1:length(buildings))
    done = false
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
                poss_coal_vals[c] = sum(min(abs.(compatible_slots.*cons_vec[c[1]]),abs.(compatible_slots.*cons_vec[c[2]])))
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