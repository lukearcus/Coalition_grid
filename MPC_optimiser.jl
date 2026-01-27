include("Buildings.jl")
struct MPC_optimiser
    energy_cost::Array{Float64}
    energy_sale::Array{Float64}
end


function optimise(opt::MPC_optimiser,bs::Vector{Building})
	num_builds = bs.size[1]
	model = Model(SCS.Optimizer)
	num_steps = length(consumption(bs[1]))
	set_silent(model)
	
	max_flow = repeat(hcat([max_store(b) for b in bs])', num_steps, 1)
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