using Plots

function plot_use(use_data::Vector{Vector{Float64}})
    x = range(1, length(use_data[1]))
    plot(x,use_data)
    savefig(string("results/",length(use_data),"_building_use_plot.pdf"))   
end