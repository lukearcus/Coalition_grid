using Plots
try
    using CSV
catch
    using CSV
end
using DataFrames
using Measures

function plot_use(use_data::Vector{Vector{Float64}}, times)
    scalefontsizes()
     default(fontfamily="Computer Modern",
        linewidth=2, framestyle=:box, label=nothing, grid=false)
    scalefontsizes(1.4)
    times = [split(split(t,"+")[1],"T")[2] for t in times]
    # x = range(1, length(use_data[1]))
    plot(times,use_data,xrotation=60,margin=5mm)
    xlabel!("Timestep")
    ylabel!("Grid trades")
    savefig(string("results/",length(use_data),"_building_use_plot.pdf"))   
end

function plot_var_builds()
        # filenames = ["results/bottom_var_num_builds.csv","results/central_ADMM_var_num_builds.csv","results/central_nonADMM_var_num_builds.csv","results/decentralised_var_num_builds.csv","results/private_var_num_builds_unguaranteed.csv"]
    filenames = ["results/bottom_var_num_builds.csv","results/central_ADMM_var_num_builds.csv","results/private_var_num_builds_unguaranteed.csv"]
    labels = ["Bottom-Up","Centralised","Private"]
    scalefontsizes()
    default(fontfamily="Computer Modern",
        linewidth=2, framestyle=:box, label=nothing, grid=false)
    scalefontsizes(1.4)
    plot()
    for (fn, lab) in zip(filenames, labels)
        file = CSV.read(fn, DataFrame, delim=",")
        plot!(file[!,"num_builds"],file[!,"num_iters"],label=lab)
    end
    xlabel!("Number of buildings")
    ylabel!("Iterations per timestep")
    
    # title!("Number of ADMM Iterations vs Number of Buildings")
    savefig(string("results/","var_builds_all.pdf"))   
end

