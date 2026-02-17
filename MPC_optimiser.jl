include("Buildings.jl")
struct MPC_optimiser
    energy_cost::Array{Float64}
    energy_sale::Array{Float64}
end

function energy_cost_k(opt::MPC_optimiser, k::Int)
	last_elem = length(opt.energy_cost)
	return vcat(opt.energy_cost[k:last_elem], opt.energy_cost[1:k-1])
end

function energy_sale_k(opt::MPC_optimiser, k::Int)
	last_elem = length(opt.energy_sale)
	return vcat(opt.energy_sale[k:last_elem], opt.energy_sale[1:k-1])
end


function optimise(opt::MPC_optimiser,bs::Vector{Building})
	num_builds = bs.size[1]
	model = Model(SCS.Optimizer)
	num_steps = length(consumption(bs[1]))
	set_silent(model)
	
	max_flow = repeat(hcat([max_flow(b) for b in bs])', num_steps, 1)
	@variable(model, 0<=pos_delta_s[t=1:num_steps, b=1:num_builds]<=max_flow[t,b])
	@variable(model, -max_flow[t,b]<=neg_delta_s[t=1:num_steps, b=1:num_builds]<=0)
	@variable(model, delta_s[1:num_steps,1:num_builds])
	@constraint(model, charge_sum, delta_s == pos_delta_s+neg_delta_s)
	
	@variable(model, grid_cons[1:num_steps, 1:num_builds] >= 0)
	@variable(model, grid_sell[1:num_steps, 1:num_builds] >= 0) # replace these with 1 variable

	@variable(model, coal_exch[1:num_steps, 1:num_builds])
	capacities = repeat(hcat([max_store(b) for b in bs])', num_steps, 1)
	#println(capacities)
	#println(capacities.size)
	@variable(model, 0 <= charge[t=1:num_steps, b=1:num_builds] <= capacities[t,b])
	@constraint(model, init_charge, charge[0,:]==[b.SoC[k] for b in bs])
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

	return model, [delta_s, grid_cons, grid_sell, charge, costs]
end

function optimise(opt::MPC_optimiser,b::Building)
	optimise(opt,[b])
end

function single_optimise(opt::MPC_optimiser,bs::Vector{MPC_Building},k::Int)
	num_builds = bs.size[1]
	model = Model(SCS.Optimizer)
	num_steps = length(pred_consumption(bs[1],k))
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

	@constraint(model, charge_c, charge .== charge_mat*charge+charge_mat*(pos_delta_s.*charge_eff_vec')+charge_mat*(neg_delta_s.*discharge_eff_vec')+init_charge_mat)

	@constraint(model, final_delta_c, delta_s[num_steps, :] >= -charge[num_steps, :])
	
	consumps = reduce(hcat, [pred_consumption(b,k) for b in bs])
	prods = reduce(hcat,[pred_production(b,k) for b in bs])
	@constraint(model, power_c, consumps+grid_sell+delta_s+coal_exch.==prods+grid_cons)
	
	@constraint(model, coal_c, coal_exch*ones(num_builds).==0)
	
	@constraint(model, cost_c, costs'.==energy_cost_k(opt,k)'*grid_cons-energy_sale_k(opt,k)'*grid_sell) # need to fix this

	@objective(model, Min, ones(num_builds)'*costs)
	
	optimize!(model)
	#println(sum(value(neg_delta_s.*pos_delta_s)))

	return model, [delta_s, grid_cons, grid_sell, charge, costs, coal_exch]
end

function single_optimise(opt::MPC_optimiser,b::MPC_Building,k::Int)
	single_optimise(opt,[b],k)
end

function optimise(opt::MPC_optimiser,bs::Vector{MPC_Building})
	num_steps = length(bs[1].act_cons)
	num_builds = length(bs)
	buy = zeros(num_steps,num_builds)
	sell = zeros(num_steps,num_builds)
	for k = 1:num_steps
		_, res = single_optimise(opt, bs, k)
		for (i, b) in enumerate(bs)
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




	if num_steps < 96
		buy_cost = opt.energy_cost[1:num_steps]
		sell_price = opt.energy_sale[1:num_steps]
	else
		buy_cost = hcat(repeat(opt.energy_cost,num_steps÷96), opt.energy_cost[1:(num_steps%96)])'
		sell_price = hcat(repeat(opt.energy_sale,num_steps÷96), opt.energy_sale[1:(num_steps%96)])'
	end
	cost = sum(buy_cost'*buy-sell_price'*sell)
	return cost, [buy, sell]
	# return single_optimise(opt, bs, 1)
end