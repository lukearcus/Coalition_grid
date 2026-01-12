include("Buildings.jl")
struct MPC_optimiser
    energy_cost::Array{Float64}
    energy_sale::Array{Float64}
end


function optimise(opt::MPC_optimiser,bs::Vector{Building})
	num_builds = bs.size[1]
	model = Model(SCS.Optimizer)
	set_silent(model)
	@variable(model, delta_s[1:24, 1:num_builds])

	
	@variable(model, grid_cons[1:24, 1:num_builds] >= 0)
	@variable(model, grid_sell[1:24, 1:num_builds] >= 0) # replace these with 1 variable

	@variable(model, coal_exch[1:24, 1:num_builds])
	@variable(model, 0 <= charge[1:24, 1:num_builds] <= 2)
	@variable(model, costs[1:num_builds])

	charge_mat = hcat(vcat(zeros(23)',I(23)),zeros(24))

	@constraint(model, charge_c, charge .== charge_mat*charge+charge_mat*delta_s)

	@constraint(model, final_delta_c, delta_s[24, :] >= -charge[24, :])
	
	consumps = reduce(hcat, [consumption(b) for b in bs])
	prods = reduce(hcat,[production(b) for b in bs])
	@constraint(model, power_c, consumps+grid_sell+delta_s+coal_exch.==prods+grid_cons)
	
	@constraint(model, coal_c, coal_exch*ones(num_builds).==0)
	
	@constraint(model, cost_c, costs'.==opt.energy_cost*grid_cons-opt.energy_sale*grid_sell)

	@objective(model, Min, ones(num_builds)'*costs)
	

	optimize!(model)
	return model, [delta_s, grid_cons, grid_sell, charge, costs]
end