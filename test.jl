using JuMP, SCS
using LinearAlgebra
using Combinatorics

include("Buildings.jl")
include("MPC_optimiser.jl")
include("Coalition.jl")
include("load_EMS_data.jl")

# energy_cost = transpose(rand(Float64, 24))
# energy_sale = 0.5*energy_cost 


num_builds = 10
#buildings = [Building((rand(Float64, 1)[1], rand(Float64, 1)[1]), rand(Float64, 24), rand(Float64, 24), rand(Float16, 1)[1],rand(Float16, 1)[1],rand(Float16, 1)[1],rand(Float16, 1)[1],i) for i = 1:num_builds]
buildings, energy_cost, energy_sale = load_from_CSV(num_builds)
opt = MPC_optimiser(energy_cost', energy_sale')


println("-------------")

t1 = time()
outs = [optimise(opt, [building]) for building in buildings]
t2 = time()

res = [out[1] for out in outs]
vars = [out[2] for out in outs]
#vars = outs[:,2]
#res2, vars2 = optimise_no_coord(buildings[2])
#res3, vars3 = optimise_no_coord(buildings[3])
#println("With storage, building 1 pays ", objective_value(res))
println("Decentralised, buildings pay ", sum([objective_value(r) for r in res]))
println("Computation took ", t2-t1,"s")
#println("delta_s ", value(vars[1]))
#println("cons ", value(vars[2]))
#println("sell ", value(vars[3]))
#println("charge ", value(vars[2][4]))
ind_coal_costs = [objective_value(r) for r in res]


println("-------------")

t3 = time()
res, vars = optimise(opt, buildings)
t4 = time()

println("Central MPC, buildings pay ", objective_value(res))
println("Computation took ", t4-t3,"s")

#println("Building 1 pays ", value(vars[5])[1])

println("-------------")

println("Coalitional optimal")

t5 = time()
opt_coal, val = find_opt_coal(buildings,3)
t6 = time()

println("In optimal coalition ",opt_coal, " agents pay ", val)
println("Computation took ", t6-t5,"s")


t7 = time()
private_coal, val = privacy_focussed_coals(buildings,3)
t8 = time()

println("In private coalition ",private_coal, " agents pay ", val)
println("Computation took ", t8-t7,"s")

