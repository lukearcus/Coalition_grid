using Convex, SCS
using Debugger

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
energy_sale = 0.5.*energy_cost 
#energy_sale = 0 .*energy_cost 
function optimise_no_coord(b::Building)
	delta_s = Variable(24)
	

	grid_cons = Variable(24)
	add_constraint!(grid_cons, grid_cons >= 0)
	grid_sell = Variable(24)
	add_constraint!(grid_sell, grid_sell >= 0)
	
	charge = Variable(24)
	add_constraint!(charge, charge >= 0)
	add_constraint!(charge, charge <= 2)
	add_constraint!(charge, charge[1] == 0)
	charge_const = [charge[i+1] == charge[i]+delta_s[i] for i in 1:23]
	
	constraints = vcat(charge_const ,[consumption(b)+grid_sell+delta_s==production(b)+grid_cons, delta_s <= 2, delta_s >= -2])
	#constraints = [consumption(b)+grid_sell+delta_s==production(b)+grid_cons, delta_s <= 2, delta_s >= -2]
	problem = minimize(energy_cost*grid_cons-energy_sale*grid_sell, constraints)
	solve!(problem, SCS.Optimizer; silent=true)
	return problem, [delta_s, grid_cons, grid_sell, charge]
end

function central_opt(bs::Vector{Building})
	num_builds = bs.size

	delta_s = Variable(24, num_builds)
	grid_cons = Variable(24, num_builds)
	add_constraint!(grid_cons, grid_cons >= 0)
	grid_sell = Variable(24, num_builds)
	add_constraint!(grid_sell, grid_sell >= 0)
	charge = Variable(24, num_builds)
	add_constraint!(charge, charge >= 0)
	add_constraint!(charge, charge[1] == 0)
	
	charge_const = [vcat([charge[i+1, b] ==charge[i, b]+delta_s for i in 1:23]) for b in 1:num_builds]
	
	constraints = vcat(charge_const ,
			   [delta_s <= 2-charge, charge <= 2, delta_s >= -charge])
	#		    , consumption(b)+grid_sell+delta_s==production(b)+grid_cons])
	#problem = minimize(sum(grid_cons.*energy_cost-grid_sell.*energy_sale), constraints)
	problem = minimize(sum(energy_cost*grid_cons-energy_sale*grid_sell), constraints)
	solve!(problem, SCS.Optimizer; silent=true)
	
	return problem
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
println("With storage, building 1 pays ", res.optval)
println("delta_s ", vars[1].value)
println("cons ", vars[2].value)
println("sell ", vars[3].value)
println("charge ", vars[4].value)
