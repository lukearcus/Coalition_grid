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
    return agents, sum(res)
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
    for coal in possible_coals
	    valid_coal = true
	    for elem in coal
		    if elem.size[1] > max_coal_size
			    valid_coal = false
			    break
		    end
	    end
	    if valid_coal
		    coal_vals[coal] = sum([objective_value(single_optimise(opt,ind_coal,k)[1]) for ind_coal in coal])
	    end
    end
    sorted_coal_vals = sort!(collect(coal_vals), by=last)
    return sorted_coal_vals[1][1]#, sorted_coal_vals[1][2]
end

function bottom_up_full_info(buildings::Vector{MPC_Building}, max_coal_size::Int, k::Int)
    agents = Vector(1:length(buildings))
    done = false
    while !done
        done = true
        #for agent in agents
        #    println(optimise(opt, buildings[agent])[1])
        #end
        outs = [single_optimise(opt, buildings[agent], k) for agent in agents]
        res = Dict([agent => out[1] for (out, agent) in zip(outs, agents)])
        vars = [out[2] for out in outs]

        poss_coals = collect(combinations(agents,2))
        poss_coal_vals = Dict()
        for c in poss_coals
            poss_coal_vec = Vector()
            append!(poss_coal_vec, c[1], c[2])
            poss_coal_vals[c] = -(objective_value(res[c[1]])+objective_value( res[c[2]]))+ sum(objective_value(single_optimise(opt, buildings[poss_coal_vec], k)[1])) #added res c[1] c[2] - joint to better discriminate
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
    return agents
end

function privacy_focussed_coals(buildings::Vector{MPC_Building}, max_coal_size::Int, k::Int)
    agents = Vector(1:length(buildings))
    done = false
    energy_diff = opt.energy_cost-opt.energy_sale
    while !done
        done = true
        #for agent in agents
        #    println(optimise(opt, buildings[agent])[1])
        #end
        outs = [single_optimise(opt, buildings[agent], k) for agent in agents]
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
    # outs = [optimise(opt, buildings[agent]) for agent in agents]
    # res = [sum(objective_value(out[1])) for out in outs]
    return agents
end

function single_coal_opt(coal_former::Function,bs::Vector{MPC_Building}, max_coal_size::Int, k::Int)
	coal = coal_former(bs,max_coal_size,k)
    # println(coal)
    # println(typeof(coal))
    if coal isa Vector{Vector{MPC_Building}}
        outs = [single_optimise(opt, agent, k) for agent in coal]
    else
        outs = [single_optimise(opt, buildings[agent], k) for agent in coal]
    end
    res = [sum(objective_value(out[1])) for out in outs]
    return coal, sum(res)
end

function coal_MPC(coal_former::Function,bs::Vector{MPC_Building}, max_coal_size::Int)
    num_steps = length(bs[1].act_cons)
	num_builds = length(bs)
	buy = zeros(num_steps,num_builds)
	sell = zeros(num_steps,num_builds)
	for k = 1:num_steps
        coal = coal_former(bs,max_coal_size,k)
        if coal isa Vector{Vector{MPC_Building}}
            outs = [single_optimise(opt, agent, k) for agent in coal]
        else
            outs = [single_optimise(opt, buildings[agent], k) for agent in coal]
        end
        res = [sum(objective_value(out[1])) for out in outs]
		# _, res = single_optimise(opt, bs, k)
		for (out,agent) in zip(outs, coal)
            if !(coal isa Vector{Vector{MPC_Building}})
                agent = buildings[agent]
            end
            res = out[2]
            for (i,b) in enumerate(agent)
                if k < num_steps
                    b.SoC[k+1] = value(res[4][2,i]) #value(res[2][1,i]-res[3][1,i]-res[6][1,i])-b.act_cons[k]+b.act_prod[k]
                end
                remaining = b.act_prod[k]-b.act_cons[k]-value(res[6][1,i])
                if remaining > 0
                    sell[k,i] = remaining
                else
                    buy[k,i]=-remaining
                end
            end
		end
	end




	if num_steps < 96
		buy_cost = opt.energy_cost[1:num_steps]
		sell_price = opt.energy_sale[1:num_steps]
	else
		buy_cost = hcat(repeat(opt.energy_cost,num_steps÷96), opt.energy_cost[1:(num_steps%96)])'
		sell_price = hcat(repeat(opt.energy_sale,num_steps÷96), opt.energy_sale[1:(num_steps%96)])'
	end
	cost = sum(buy_cost'*buy-sell_price'*sell)
	return cost, [buy, sell]
end