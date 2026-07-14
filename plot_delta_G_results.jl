#!/usr/bin/env julia

# Script to plot the results of delta_G experiment
using CSV, DataFrames, Plots
using Statistics

# Read the results
data = CSV.read("results/private_delta_G_test.csv", DataFrame)

# Group data by delta_G and calculate mean cost and std deviation
grouped = combine(groupby(data, :delta_G), :average_cost => mean => :mean_cost, :average_cost => std => :std_cost)

# Sort by delta_G
sort!(grouped, :delta_G)

# Create the plot
plt = plot(grouped.delta_G, grouped.mean_cost, 
    yerror=grouped.std_cost,
    seriestype=:scatter,
    xlabel="delta_G",
    ylabel="Average Cost",
    title="Cost vs delta_G in Private Coalition Formation",
    markersize=6,
    legend=false)

# Add a line connecting the points
plot!(grouped.delta_G, grouped.mean_cost, 
    seriestype=:line,
    linewidth=2,
    color=:blue)

# Save the plot
savefig(plt, "results/delta_G_cost_plot.pdf")

println("Plot saved to results/delta_G_cost_plot.pdf")