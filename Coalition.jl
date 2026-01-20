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
    return sorted_coal_vals[1]
end

function privacy_focussed_coals(buildings::Vector{Building}, max_coal_size::Int)
    outs = [optimise(opt, [building]) for building in buildings]
    res = [out[1] for out in outs]
    vars = [out[2] for out in outs]

    cons_vec = [vec(value(var[2]-var[3])) for var in vars]

    agents = Vector(1:length(buildings))
    done = false
    while !done
        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        for c in poss_coals
            if any(cons_vec[c[1]].*cons_vec[c[2]] < zeros(length(cons_vec[c[1]])))
                poss_coal_vals[c] = norm(cons_vec[c[1]]+cons_vec[c[2]])
            else
                poss_coal_vals[c] = 9999
            end
        end
        sorted_coal_vals = sort!(collect(poss_coal_vals), by=last)
        if sorted_coal_vals[1] == 9999
            break
        end
        new_agents = Vector()
        for elem in sorted_coal_vals
            if !(elem[1][1] in new_agents) && !(elem[1][2] in new_agents)
                push!(new_agents, elem[1])
            end
        end
        println(sorted_coal_vals)
    end
    #println(cons_vec)
    #println("cons ", value(vars[2][2]))
    #println("sell ", vars[1][3])
end