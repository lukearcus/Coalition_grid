try
    using CSV
catch
    using CSV
end
@eval CSV.Parsers import Base.Ryu: writeshortest
using DataFrames
include("../Buildings.jl")
include("../MPC_optimiser.jl")
include("../Coalition.jl")
include("../load_EMS_data.jl")
include("../plotting.jl")

num_builds = 20
max_coal_size = 1
num_steps = 96
num_ahead = 8

#buildings = [Building((rand(Float64, 1)[1], rand(Float64, 1)[1]), rand(Float64, 24), rand(Float64, 24), rand(Float16, 1)[1],rand(Float16, 1)[1],rand(Float16, 1)[1],rand(Float16, 1)[1],i) for i = 1:num_builds]
buildings, energy_cost, energy_sale = MPC_load_from_CSV(num_builds,num_steps)

opt = MPC_optimiser(energy_cost', energy_sale')

# println("test")

# res, _, num_iters = coal_MPC(privacy_focussed_coals,buildings,max_coal_size)

data = DataFrame(max_size=[],average_cost=[],num_iters=[],time=[])

for max_coal_size in 2:num_builds
    try
        t1 = time()
        res, _, num_iters = coal_MPC(find_opt_coal,buildings,max_coal_size, num_ahead)
        t2 = time()

        runtime= t2-t1
        push!(data, [max_coal_size,res/num_builds,num_iters,runtime])
        CSV.write("results/opt_var_size.csv", data)
    catch
        println("Failed at", max_coal_size)
    end
end