using Plots
try
    using CSV
catch
    using CSV
end
using DataFrames

function plot_use(use_data::Vector{Vector{Float64}})
    x = range(1, length(use_data[1]))
    plot(x,use_data)
    xlabel!("Timestep")
    ylabel!("Grid trades (positive indicates purchases from grid)")
    savefig(string("results/",length(use_data),"_building_use_plot.pdf"))   
end

function plot_var_builds()
        # filenames = ["results/bottom_var_num_builds.csv","results/central_ADMM_var_num_builds.csv","results/central_nonADMM_var_num_builds.csv","results/decentralised_var_num_builds.csv","results/private_var_num_builds_unguaranteed.csv"]
    filenames = ["results/bottom_var_num_builds.csv","results/central_ADMM_var_num_builds.csv","results/private_var_num_builds_unguaranteed.csv"]
    labels = ["Bottom-Up Coalition Formation","Centralised Solution","Private Coalition Formation"]
    plot()
    for (fn, lab) in zip(filenames, labels)
        file = CSV.read(fn, DataFrame, delim=",")
        plot!(file[!,"num_builds"],file[!,"num_iters"],label=lab)
    end
    xlabel!("Number of buildings")
    ylabel!("Average number of iterations per timestep")
    # title!("Number of ADMM Iterations vs Number of Buildings")
    savefig(string("results/","var_builds_all.pdf"))   
end

