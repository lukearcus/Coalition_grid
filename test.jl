using JuMP, SCS
#using Convex, SCS
using Debugger
using LinearAlgebra

abstract type Player end

struct Building <: Player
	loc::Tuple
	cons::Vector{Float64}
	prod::Vector{Float64}
	max_storage::Float16
end

location(b::Building) = b.loc
consumption(b::Building) = b.cons
production(b::Building) = b.prod

energy_cost = transpose(rand(Float64, 24))
energy_sale = 0.5*energy_cost 

coal_cost = (energy_cost+energy_sale)/2
#energy_sale = 0 .*energy_cost 
function optimise_no_coord(b::Building)
	model = Model(SCS.Optimizer)
	@variable(model, delta_s[1:24])

	
	@variable(model, grid_cons[1:24] >= 0)
	@variable(model, grid_sell[1:24] >= 0)
	
	@variable(model, 0 <= charge[1:24] <= 2)
	charge_mat = hcat(vcat(zeros(23)',I(23)),zeros(24))
	@constraint(model, charge_c, charge .== charge_mat*charge+charge_mat*delta_s)

	@constraint(model, final_delta_c, delta_s[24] >= -charge[24])
	
	@constraint(model, power_c, consumption(b)+grid_sell+delta_s.==production(b)+grid_cons)
	@expression(model, buy, energy_cost*grid_cons)
	@expression(model, sell, energy_sale*grid_sell)
	@objective(model, Min, buy-sell)
	optimize!(model)
	return model, [delta_s, grid_cons, grid_sell, charge]
end

function central_opt(bs::Vector{Building})
	num_builds = bs.size[1]
	model = Model(SCS.Optimizer)
	@variable(model, delta_s[1:24, 1:num_builds])

	
	@variable(model, grid_cons[1:24, 1:num_builds] >= 0)
	@variable(model, grid_sell[1:24, 1:num_builds] >= 0)

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
	
	@constraint(model, cost_c, costs'.==energy_cost*grid_cons-energy_sale*grid_sell-coal_cost*coal_exch)

	@objective(model, Min, ones(num_builds)'*costs)
	

	optimize!(model)
	return model, [delta_s, grid_cons, grid_sell, charge, costs]
end

#rand_num = 
#println(rand_num)
num_builds = 3
buildings = [Building((rand(Float64, 1)[1], rand(Float64, 1)[1]), rand(Float64, 24), rand(Float64, 24), rand(Float16, 1)[1] ) for i = 1:num_builds]

#build_1 = Building((0.0,0.0),[1,2,3],[3,2,1],2)

println(energy_cost*consumption(buildings[1]))
println(energy_sale*production(buildings[1]))

#no_opt_cost = [energy_cost*consumption(b)-energy_sale*production(b) for b in buildings]
req_energy = [consumption(b)-production(b) for b in buildings]
energy_cons = [[elem >= 0 ? elem : 0 for elem in req_en] for req_en in req_energy]
energy_sold = [[elem <= 0 ? -elem : 0 for elem in req_en] for req_en in req_energy]
no_opt_cost = [sum(energy_cost*en_cons-(energy_sale*en_sold)[1]) for (en_cons, en_sold) in zip(energy_cons, energy_sold)]

println("Costs: ", no_opt_cost)

println("Building 1 is in location ", buildings[1].loc)
println("Building 1 requires energy ", consumption(buildings[1]))
println("Building 1 produces energy ", production(buildings[1]))
println("Building 1 pays ", no_opt_cost[1])

println("-------------")

res, vars = optimise_no_coord(buildings[1])
res2, vars2 = optimise_no_coord(buildings[2])
res3, vars3 = optimise_no_coord(buildings[3])
println("With storage, building 1 pays ", objective_value(res))
println("Decentralised, buildings pay ", objective_value(res)+objective_value(res2)+objective_value(res3))
println("delta_s ", value(vars[1]))
println("cons ", value(vars[2]))
println("sell ", value(vars[3]))
println("charge ", value(vars[4]))


println("-------------")

res, vars = central_opt(buildings)
println("Cental MPC, buildings pay ", objective_value(res))
println("Building 1 pays ", value(vars[5])[1])
println("delta_s ", value(vars[1]))
println("cons ", value(vars[2]))
println("sell ", value(vars[3]))
println("charge ", value(vars[4]))
