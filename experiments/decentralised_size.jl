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

num_builds = 96
max_coal_size = 1
num_steps = 96

#buildings = [Building((rand(Float64, 1)[1], rand(Float64, 1)[1]), rand(Float64, 24), rand(Float64, 24), rand(Float16, 1)[1],rand(Float16, 1)[1],rand(Float16, 1)[1],rand(Float16, 1)[1],i) for i = 1:num_builds]
buildings, energy_cost, energy_sale = load_from_CSV(num_builds,num_steps)

opt = MPC_optimiser(energy_cost', energy_sale')

# println("test")

# res, _, num_iters = coal_MPC(privacy_focussed_coals,buildings,max_coal_size)

data = DataFrame(max_coal=[],average_cost=[],num_iters=[],time=[])

for max_coal_size in 2:70
    t1 = time()
    outs = [optimise(opt, [building]) for building in buildings]
    t2 = time()

    res = [out[1] for out in outs]
    vars = [out[2] for out in outs]
    num_iters = sum([out[3] for out in outs])

    val = sum([r for r in res])
    runtime= t2-t1
    push!(data, [max_coal_size,val/num_builds,num_iters,runtime])
    CSV.write("results/decentralised_var_size.csv", data)
end