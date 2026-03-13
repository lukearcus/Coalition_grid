include("Buildings.jl")
include("MPC_optimiser.jl")
include("Coalition.jl")
include("load_EMS_data.jl")
include("plotting.jl")

# energy_cost = transpose(rand(Float64, 24))
# energy_sale = 0.5*energy_cost 


num_builds = 2
max_coal_size = 2
num_steps = 1

#buildings = [Building((rand(Float64, 1)[1], rand(Float64, 1)[1]), rand(Float64, 24), rand(Float64, 24), rand(Float16, 1)[1],rand(Float16, 1)[1],rand(Float16, 1)[1],rand(Float16, 1)[1],i) for i = 1:num_builds]
buildings, energy_cost, energy_sale = load_from_CSV(num_builds,num_steps)
opt = MPC_optimiser(energy_cost', energy_sale')

use_data = [b.act_cons-b.act_prod for b in buildings]
plot_use(use_data)

# println("test")

# res, _, num_iters = coal_MPC(privacy_focussed_coals,buildings,max_coal_size)

println("-------------")

t1 = time()
outs = [optimise(opt, [building]) for building in buildings]
t2 = time()

res = [out[1] for out in outs]
vars = [out[2] for out in outs]
num_iters = sum([out[3] for out in outs])
#vars = outs[:,2]
#res2, vars2 = optimise_no_coord(buildings[2])
#res3, vars3 = optimise_no_coord(buildings[3])
#println("With storage, building 1 pays ", objective_value(res))
println("Decentralised, buildings pay ", sum([r for r in res]))
println("Computation took ", t2-t1,"s")
println("Required average of ", num_iters, " iterations")
#println("delta_s ", value(vars[1]))
#println("cons ", value(vars[2]))
#println("sell ", value(vars[3]))
#println("charge ", value(vars[2][4]))
#ind_coal_costs = [objective_value(r) for r in res]


println("-------------")

t3 = time()
res, vars, num_iters = optimise(opt, buildings, false)
t4 = time()

println("Central MPC, buildings pay ", res)
println("Computation took ", t4-t3,"s")
println("Required average of ", num_iters, " iterations")


t3 = time()
res, vars, num_iters = optimise(opt, buildings)
t4 = time()

println("Central MPC, ADMM, buildings pay ", res)
println("Computation took ", t4-t3,"s")
println("Required average of ", num_iters, " iterations")


#println("Building 1 pays ", value(vars[5])[1])

println("-------------")


t7 = time()
# private_coal = privacy_focussed_coals(buildings,max_coal_size,1)
# private_coal, res = single_coal_opt(privacy_focussed_coals,buildings,max_coal_size,1)
res, _, num_iters = coal_MPC(privacy_focussed_coals,buildings,max_coal_size)

t8 = time()

# println("In private coalition ",private_coal, " agents pay ", res)
println("In private coalitions agents pay ", res)
println("Computation took ", t8-t7,"s")
println("Required average of ", num_iters, " iterations")


println("-------------")

t9 = time()
# private_coal = bottom_up_full_info(buildings,max_coal_size,1)
# bottom_coal, res = single_coal_opt(bottom_up_full_info,buildings,max_coal_size,1)
res, _, num_iters = coal_MPC(bottom_up_full_info,buildings,max_coal_size)

t10 = time()

# println("In full-info bottom-up coalition ",bottom_coal, " agents pay ", res)
println("In full-info bottom-up coalition agents pay ", res)
println("Computation took ", t10-t9,"s")
println("Required average of ", num_iters, " iterations")

println("-------------")


if num_builds <= 8 #takes about 20 mins on laptop to execute 8
    println("Coalitional optimal")

    t5 = time()
    # opt_coal, res = single_coal_opt(find_opt_coal,buildings,max_coal_size,1)
    res, _, num_iters = coal_MPC(find_opt_coal,buildings,max_coal_size)

    # opt_coal = find_opt_coal(buildings,max_coal_size,1)
    t6 = time()

    # println("In optimal coalition ",opt_coal, " agents pay ", res)
    println("In optimal coalition agents pay ", res)
    println("Computation took ", t6-t5,"s")
    println("Required average of ", num_iters, " iterations")

end


