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

    cons_vec = [value(var[2]-var[3]) for var in vars]

    
    #println(cons_vec)
    #println("cons ", value(vars[2][2]))
    #println("sell ", vars[1][3])
end