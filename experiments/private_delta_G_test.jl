try
    using CSV
catch
    using CSV
end
@eval CSV.Parsers import Base.Ryu: writeshortest
using DataFrames
using Random
include("../Buildings.jl")
include("../MPC_optimiser.jl")
include("../Coalition.jl")
include("../load_EMS_data.jl")
include("../plotting.jl")

num_builds = 10
max_coal_size = 6
num_steps = 96
num_ahead = 8

# Load building data
all_buildings, energy_cost, energy_sale = MPC_load_from_CSV(num_builds, num_steps)
opt = MPC_optimiser(energy_cost', energy_sale')

# Test delta_G values from 0.01 to 1000 in increments of about 100 (10 values total) 
delta_G_values = range(0.01, 500, length=100)
num_repeats = 10

# Initialize results DataFrame
data = DataFrame(delta_G=[], repeat=[], average_cost=[], num_iters=[], time=[])

for (delta_G_idx, delta_G) in enumerate(delta_G_values)
    println("Testing delta_G = $delta_G")
    for repeat in 1:num_repeats
        try
            # Set seed for reproducibility of each repeat
            Random.seed!(repeat + delta_G_idx * 1000)
            
            buildings = all_buildings[1:num_builds]
            t1 = time()
            
            # Use the modified function with delta_G parameter
            res, _, num_iters = coal_MPC((buildings, max_coal_size, k, num_ahead, receding_horizon) -> 
                privacy_focussed_coals_with_delta(buildings, max_coal_size, k, num_ahead, receding_horizon, delta_G), 
                buildings, max_coal_size, num_ahead)
            
            t2 = time()
            
            runtime = t2 - t1
            push!(data, [delta_G, repeat, res/num_builds, num_iters, runtime])
            
            # Save results incrementally
            CSV.write("results/private_delta_G_test.csv", data)
            
        catch e
            println("Failed at delta_G=$delta_G, repeat=$repeat")
            # Print stacktrace for debugging
            println(e)
        end
    end
end

println("Experiment completed. Results saved to results/private_delta_G_test.csv")