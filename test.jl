using JuMP, SCS
using LinearAlgebra
using Combinatorics

include("Buildings.jl")
include("MPC_optimiser.jl")
include("Coalition.jl")

energy_cost = transpose(rand(Float64, 24))
energy_sale = 0.5*energy_cost 

opt = MPC_optimiser(energy_cost, energy_sale)

num_builds = 5
buildings = [Building((rand(Float64, 1)[1], rand(Float64, 1)[1]), rand(Float64, 24), rand(Float64, 24), rand(Float16, 1)[1],i) for i = 1:num_builds]

privacy_focussed_coals(buildings,3)

println("-------------")

outs = [optimise(opt, [building]) for building in buildings]
res = [out[1] for out in outs]
vars = [out[2] for out in outs]
#vars = outs[:,2]
#res2, vars2 = optimise_no_coord(buildings[2])
#res3, vars3 = optimise_no_coord(buildings[3])
#println("With storage, building 1 pays ", objective_value(res))
println("Decentralised, buildings pay ", sum([objective_value(r) for r in res]))
#println("delta_s ", value(vars[1]))
#println("cons ", value(vars[2]))
#println("sell ", value(vars[3]))
#println("charge ", value(vars[4]))
ind_coal_costs = [objective_value(r) for r in res]


println("-------------")

res, vars = optimise(opt, buildings)
println("Central MPC, buildings pay ", objective_value(res))
#println("Building 1 pays ", value(vars[5])[1])

println("-------------")

println("Coalitional optimal")

opt_coal = find_opt_coal(buildings,3)
println(opt_coal)

privacy_focussed_coals(buildings,3)