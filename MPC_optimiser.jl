using JuMP, SCS
using LinearAlgebra

include("Buildings.jl")
struct MPC_optimiser
    energy_cost::Array{Float64}
    energy_sale::Array{Float64}
end

function energy_cost_k(opt::MPC_optimiser, k::Int, num_steps::Int=96)
	last_elem = length(opt.energy_cost)
	return vcat(opt.energy_cost[k:last_elem], opt.energy_cost[1:k-1])[1:num_steps]
end

function energy_sale_k(opt::MPC_optimiser, k::Int, num_steps::Int=96)
	last_elem = length(opt.energy_sale)
	return vcat(opt.energy_sale[k:last_elem], opt.energy_sale[1:k-1])[1:num_steps]
end


function optimise(opt::MPC_optimiser,bs::Vector{Building},ADMM::Bool=true,receding_horizon::Bool=false)
	num_builds = bs.size[1]
	model = Model(SCS.Optimizer)
	num_steps = length(consumption(bs[1]))
	set_silent(model)
	
	max_flow_val = repeat(hcat([max_flow(b) for b in bs])', num_steps, 1)
	@variable(model, 0<=pos_delta_s[t=1:num_steps, b=1:num_builds]<=max_flow_val[t,b])
	@variable(model, -max_flow_val[t,b]<=neg_delta_s[t=1:num_steps, b=1:num_builds]<=0)
	@variable(model, delta_s[1:num_steps,1:num_builds])
	@constraint(model, charge_sum, delta_s == pos_delta_s+neg_delta_s)
	
	@variable(model, grid_cons[1:num_steps, 1:num_builds] >= 0)
	@variable(model, grid_sell[1:num_steps, 1:num_builds] >= 0) # replace these with 1 variable

	@variable(model, coal_exch[1:num_steps, 1:num_builds])
	capacities = repeat(hcat([max_store(b) for b in bs])', num_steps, 1)
	#println(capacities)
	#println(capacities.size)
	@variable(model, 0 <= charge[t=1:num_steps, b=1:num_builds] <= capacities[t,b])
	@constraint(model, init_charge, charge[1,:]==[0 for b in bs])
	@variable(model, costs[1:num_builds])

	charge_mat = hcat(vcat(zeros(num_steps-1)',I(num_steps-1)),zeros(num_steps))

	#println(charge_eff_vec)

	charge_eff_vec = hcat([charge_eff(b) for b in bs])
	discharge_eff_vec = hcat([1/discharge_eff(b) for b in bs])
	@constraint(model, charge_c, charge .== charge_mat*charge+charge_mat*(pos_delta_s.*charge_eff_vec')+charge_mat*(neg_delta_s.*discharge_eff_vec'))

	@constraint(model, final_delta_c, delta_s[num_steps, :] >= -charge[num_steps, :])
	
	consumps = reduce(hcat, [consumption(b) for b in bs])
	prods = reduce(hcat,[production(b) for b in bs])
	@constraint(model, power_c, consumps+grid_sell+delta_s+coal_exch.==prods+grid_cons)
	
	@constraint(model, coal_c, coal_exch*ones(num_builds).==0)
	
	@constraint(model, cost_c, costs'.==opt.energy_cost*grid_cons-opt.energy_sale*grid_sell)

	@objective(model, Min, ones(num_builds)'*costs)
	
	optimize!(model)
	#println(sum(value(neg_delta_s.*pos_delta_s)))

	return sum(value(costs)), [delta_s, grid_cons, grid_sell, charge, costs], 1
end

function optimise(opt::MPC_optimiser,b::Building,ADMM::Bool=true,receding_horizon::Bool=false)
	return optimise(opt,[b])
end

function single_optimise(opt::MPC_optimiser,bs::Vector{MPC_Building},k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
	num_builds = bs.size[1]
	model = Model(SCS.Optimizer)
	num_steps = num_look_ahead
	if !receding_horizon
		num_steps = min(length(bs[1].act_cons)-k+1,num_steps)
	end
	num_steps = min(length(pred_consumption(bs[1],k)),num_steps)
	num_steps = max(num_steps, 1)
	set_silent(model)
	
	max_flow_val = repeat(hcat([max_flow(b) for b in bs])', num_steps, 1)
	@variable(model, 0<=pos_delta_s[t=1:num_steps, b=1:num_builds]<=max_flow_val[t,b])
	@variable(model, -max_flow_val[t,b]<=neg_delta_s[t=1:num_steps, b=1:num_builds]<=0)
	@variable(model, delta_s[1:num_steps,1:num_builds])
	@constraint(model, charge_sum, delta_s == pos_delta_s+neg_delta_s)
	
	@variable(model, grid_cons[1:num_steps, 1:num_builds] >= 0)
	@variable(model, grid_sell[1:num_steps, 1:num_builds] >= 0) # replace these with 1 variable

	@variable(model, coal_exch[1:num_steps, 1:num_builds])
	capacities = repeat(hcat([max_store(b) for b in bs])', num_steps, 1)
	#println(capacities)
	#println(capacities.size)
	@variable(model, 0 <= charge[t=1:num_steps, b=1:num_builds] <= capacities[t,b])
	@variable(model, costs[1:num_builds])

	charge_mat = hcat(vcat(zeros(num_steps-1)',I(num_steps-1)),zeros(num_steps))

	#println(charge_eff_vec)

	charge_eff_vec = hcat([charge_eff(b) for b in bs])
	discharge_eff_vec = hcat([1/discharge_eff(b) for b in bs])

	init_charge_mat = vcat([b.SoC[k] for b in bs]',zeros(num_steps-1,num_builds))
	# println([b.SoC[k] for b in bs])
	# if isnan(bs[1].SoC[k][1])
	# 	b=bs[1]
	# 	println(b)
	# 	println(b.SoC)
	# 	println(k)
	# end
	# println(charge_mat)
	#println([b.SoC[k] for b in bs])
	#println(k)
	# println(init_charge_mat)
	@constraint(model, charge_c, charge .== charge_mat*charge+charge_mat*(pos_delta_s.*charge_eff_vec')+charge_mat*(neg_delta_s.*discharge_eff_vec')+init_charge_mat)

	@constraint(model, final_delta_c, delta_s[num_steps, :] >= -charge[num_steps, :])
	
	consumps = reduce(hcat, [pred_consumption(b,k,num_steps) for b in bs])
	prods = reduce(hcat,[pred_production(b,k,num_steps) for b in bs])
	@constraint(model, power_c, consumps+grid_sell+delta_s+coal_exch.==prods+grid_cons)
	
	@constraint(model, coal_c, coal_exch*ones(num_builds).==0)
	
	@constraint(model, cost_c, costs'.==energy_cost_k(opt,k,num_steps)'*grid_cons-energy_sale_k(opt,k,num_steps)'*grid_sell) # need to fix this

	@objective(model, Min, ones(num_builds)'*costs)
	
	optimize!(model)
	#println(sum(value(neg_delta_s.*pos_delta_s)))

	return model, [delta_s, grid_cons, grid_sell, charge, costs, coal_exch]
end

function ADMM_build_opt(b::MPC_Building,coal_trades::Vector{Float64},lambda::Vector{Float64},k::Int,c::Float64)
	model = Model(SCS.Optimizer)
	num_steps = length(lambda)
	set_silent(model)
	
	max_flow_val = ones(num_steps).*max_flow(b)
	@variable(model, 0<=pos_delta_s[t=1:num_steps]<=max_flow_val[t])
	@variable(model, -max_flow_val[t]<=neg_delta_s[t=1:num_steps]<=0)
	@variable(model, delta_s[1:num_steps])
	@constraint(model, charge_sum, delta_s == pos_delta_s+neg_delta_s)
	
	@variable(model, grid_cons[1:num_steps] >= 0)
	@variable(model, grid_sell[1:num_steps] >= 0) # replace these with 1 variable

	@variable(model, coal_exch[1:num_steps])
	capacities = ones(num_steps).*max_store(b)
	#println(capacities)
	#println(capacities.size)
	@variable(model, 0 <= charge[t=1:num_steps] <= capacities[t])
	@variable(model, cost)

	charge_mat = hcat(vcat(zeros(num_steps-1)',I(num_steps-1)),zeros(num_steps))

	#println(charge_eff_vec)

	charge_eff_vec = charge_eff(b)'
	discharge_eff_vec = 1/discharge_eff(b)'

	init_charge_mat = vcat(b.SoC[k]',zeros(num_steps-1,1))

	@constraint(model, charge_c, charge .== charge_mat*charge+charge_mat*(pos_delta_s.*charge_eff_vec')+charge_mat*(neg_delta_s.*discharge_eff_vec')+init_charge_mat)

	@constraint(model, final_delta_c, delta_s[num_steps, :] >= -charge[num_steps, :])
	
	consumps =  pred_consumption(b,k,num_steps)
	prods =pred_production(b,k,num_steps)
	@constraint(model, power_c, consumps+grid_sell+delta_s+coal_exch.==prods+grid_cons)
	
	# @constraint(model, coal_c, coal_exch*ones(num_builds).==0)
	
	@constraint(model, cost_c, cost==energy_cost_k(opt,k,num_steps)'*grid_cons-energy_sale_k(opt,k,num_steps)'*grid_sell) # need to fix this

	@objective(model, Min, cost+lambda'*coal_exch+(c/2)*(coal_exch-coal_trades)'*(coal_exch-coal_trades))
	
	optimize!(model)
	#println(sum(value(neg_delta_s.*pos_delta_s)))
	return model, [delta_s, grid_cons, grid_sell, charge, cost, coal_exch]
end

function ADMM_coal_update(lambda::Vector{Vector{Float64}},proposed_coals::Vector{Vector{Float64}},c::Float64)
	model = Model(SCS.Optimizer)
	num_builds = length(lambda)
	num_steps = length(lambda[1])
	set_silent(model)
	@variable(model, coal_exch[1:num_steps,1:num_builds])
	@constraint(model, coal_c, coal_exch*ones(num_builds).==0)

	proposed_coals = reduce(hcat, proposed_coals)
	lambda = reduce(hcat, lambda)
	@objective(model, Min,-sum(lambda.*coal_exch)+(c/2)*sum((proposed_coals-coal_exch).*(proposed_coals-coal_exch)))

	optimize!(model)

	return model, coal_exch
end

function single_optimise_ADMM(opt::MPC_optimiser,bs::Vector{MPC_Building},k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
	num_builds = bs.size[1]
	if num_builds == 1
		return single_optimise(opt,bs,k,num_look_ahead,receding_horizon)[2], 1
	end
	if !receding_horizon
		num_steps = min(length(bs[1].act_cons)-k+1, num_look_ahead)
	else
		num_steps = num_look_ahead
	end
	num_steps = min(length(pred_consumption(bs[1],k)),num_steps)
	num_steps = max(num_steps, 1)
	c=0.5
	new_coal = zeros(num_steps,num_builds)
	proposed_coal =[zeros(num_steps) for i in 1:num_builds]
	lambdas = [zeros(num_steps) for i in 1:num_builds]
	num_iters_admm = 1
	# need a solution in case not converged (use new_coal)
	poss_sol = proposed_coal
	let states = Vector{Any}(undef,num_builds)
		while (norm(reduce(hcat, proposed_coal).-new_coal) > 1e-5) | (num_iters_admm <= 1)
			# c *= 1.1
			res = Vector{Any}(undef,num_builds)
			Threads.@threads for ind in 1:num_builds
				res[ind] = ADMM_build_opt(bs[ind],new_coal[:,ind],lambdas[ind],k,c)
			end
			# res = [ADMM_build_opt(b,new_coal[:,ind],lambda,k,c) for ( b, lambda, ind) in zip(bs,lambdas, 1:num_builds)]
			states = [r[2] for r in res]
			proposed_coal = [value(state[6]) for state in states]

			model, new_coal = ADMM_coal_update(lambdas,proposed_coal,c)
			new_coal = value(new_coal)
			if !any(isnan,new_coal)
				poss_sol = new_coal
			end

			lambdas = [lambda + c*(prop_coal-new_coal[:,ind]) for (lambda, prop_coal, ind) in zip(lambdas, proposed_coal, 1:num_builds)]
			num_iters_admm += 1
			# if num_iters_admm > 1000
			# 	println("Convergence failed")
			# 	break
			# end
			# println(proposed_coal)
			# z update
			#lambda update
		end
		# if num_iters_admm < 1000
		# 	println("Converged in limit")
		# end
		for state in states
			state[6] = poss_sol
		end
		states = [reduce(hcat,[state[i] for state in states]) for i in 1:length(states[1])]
		return states, num_iters_admm
	end	
end

function single_optimise_ADMM(opt::MPC_optimiser,b::MPC_Building,k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
	return single_optimise_ADMM(opt,[b],k,num_look_ahead,receding_horizon)
end
function single_optimise(opt::MPC_optimiser,b::MPC_Building,k::Int,num_look_ahead::Int,receding_horizon::Bool=false)
	return single_optimise(opt,[b],k,num_look_ahead,receding_horizon)
end

function optimise(opt::MPC_optimiser,bs::Vector{MPC_Building},num_look_ahead::Int,ADMM::Bool=true,receding_horizon::Bool=false)
	num_steps = length(bs[1].act_cons)
	num_builds = length(bs)
	buy = zeros(num_steps,num_builds)
	sell = zeros(num_steps,num_builds)
	num_iters = 0
	for k = 1:num_steps
		# res = single_optimise_ADMM(opt, bs, k,true)
		# println(sum(value(res[5])))
		# model, res = single_optimise(opt, bs, k,true)
		# println(sum(value(res[5])))

		if ADMM
			res, num_iters_k = single_optimise_ADMM(opt, bs, k,num_look_ahead,receding_horizon)
			num_iters += num_iters_k
		else
			model, res = single_optimise(opt, bs, k,num_look_ahead,receding_horizon)
			num_iters += 1
		end
		for (i, b) in enumerate(bs)
			if k < num_steps
				b.SoC[k+1] = max(0,value(res[4][2,i])) #value(res[2][1,i]-res[3][1,i]-res[6][1,i])-b.act_cons[k]+b.act_prod[k]
				#println(b.SoC[k+1])
			end
			remaining = b.act_prod[k]-b.act_cons[k]-value(res[6][1,i]+res[1][1,i])
			if remaining > 0
				sell[k,i] = remaining
			else
				buy[k,i]=-remaining
			end		
		end
	end
	if num_steps <= 96
		buy_cost = opt.energy_cost[1:num_steps]
		sell_price = opt.energy_sale[1:num_steps]
	else
		buy_cost = hcat(repeat(opt.energy_cost,num_steps÷96), opt.energy_cost[1:(num_steps%96)])'
		sell_price = hcat(repeat(opt.energy_sale,num_steps÷96), opt.energy_sale[1:(num_steps%96)])'
	end
	cost = sum(buy_cost'*buy-sell_price'*sell)
	return cost, [buy, sell], num_iters/num_steps
	# return single_optimise(opt, bs, 1)
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

# function enumerate(itt::MPC_Building)
# 	return 1, itt
# end

function coal_MPC(coal_former::Function,bs::Vector{MPC_Building}, max_coal_size::Int,num_look_ahead::Int,receding_horizon::Bool=false)
    num_steps = length(bs[1].act_cons)
	num_builds = length(bs)
	buy = zeros(num_steps,num_builds)
	sell = zeros(num_steps,num_builds)
	num_iters = 0
	for k = 1:num_steps
        coal, outs, num_iters_k = coal_former(bs,max_coal_size,k,num_look_ahead,receding_horizon)
		num_iters += num_iters_k
        # if coal isa Vector{Vector{MPC_Building}}
        #     outs = [single_optimise_ADMM(opt, agent, k) for agent in coal]
        # else
        #     outs = [single_optimise_ADMM(opt, buildings[agent], k) for agent in coal]
        # end
        #res = [sum(value(out[5])) for out in outs]
		# _, res = single_optimise(opt, bs, k)
		for (res,agent) in zip(outs, coal)
            if !(coal isa Vector{Vector{MPC_Building}})
                agent = bs[agent]
            end
			if !(agent isa MPC_Building)
				for (i,b) in enumerate(agent)
					if k < num_steps
						b.SoC[k+1] = max(0,value(res[4][2,i])) #value(res[2][1,i]-res[3][1,i]-res[6][1,i])-b.act_cons[k]+b.act_prod[k]
					end
					# println(b.SoC)
					remaining = b.act_prod[k]-b.act_cons[k]-value(res[6][1,i]+
					res[1][1,i])
					if remaining > 0
						sell[k,b.id] = remaining
					else
						buy[k,b.id]=-remaining
					end
					# println(remaining)
				end
			else
				b = agent
				if k < num_steps
					b.SoC[k+1] = max(0,value(res[4][2,1])) #value(res[2][1,i]-res[3][1,i]-res[6][1,i])-b.act_cons[k]+b.act_prod[k]
				end
				remaining = b.act_prod[k]-b.act_cons[k]-value(res[6][1,1]+res[1][1,1])
				if remaining > 0
					sell[k,b.id] = remaining
				else
					buy[k,b.id]=-remaining
				end
			end
		end
	end
	if num_steps <= 96
		buy_cost = opt.energy_cost[1:num_steps]
		sell_price = opt.energy_sale[1:num_steps]
	else
		buy_cost = hcat(repeat(opt.energy_cost,num_steps÷96), opt.energy_cost[1:(num_steps%96)])'
		sell_price = hcat(repeat(opt.energy_sale,num_steps÷96), opt.energy_sale[1:(num_steps%96)])'
	end
	cost = sum(buy_cost'*buy-sell_price'*sell)
	return cost, [buy, sell], num_iters/num_steps
end