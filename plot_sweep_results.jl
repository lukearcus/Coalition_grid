using CSV, DataFrames, Plots, Statistics

# Read the CSV file
df = CSV.read(joinpath(@__DIR__, "results", "sweep_delta_G_subset.csv"), DataFrame)

# Set plotting backend
gr()

# Filter out delta_G == 0 (invalid for log scale)
df_log = filter(:delta_G => x -> x > 0, df)

# Compute mean of each metric per delta_G value (across repeats)
gdf = combine(groupby(df_log, :delta_G),
              :average_cost  => mean => :mean_cost,
              :benefit_vs_dec => mean => :mean_benefit,
              :time          => mean => :mean_time,
              :num_iters     => mean => :mean_iters)
sort!(gdf, :delta_G)

# Plot 1: Average cost vs delta_G
p1 = plot(df_log[:, :delta_G], df_log[:, :average_cost],
          seriestype=:scatter, xscale=:log10,
          xlabel="delta_G", ylabel="Average Cost",
          title="Average Cost vs delta_G", legend=false)
plot!(p1, gdf[:, :delta_G], gdf[:, :mean_cost],
      seriestype=:line, linewidth=2, color=:red, label="mean")

# Plot 2: Benefit vs delta_G
p2 = plot(df_log[:, :delta_G], df_log[:, :benefit_vs_dec],
          seriestype=:scatter, xscale=:log10,
          xlabel="delta_G", ylabel="Benefit vs Decentralized",
          title="Benefit vs Decentralized vs delta_G", legend=false)
plot!(p2, gdf[:, :delta_G], gdf[:, :mean_benefit],
      seriestype=:line, linewidth=2, color=:red, label="mean")

# Plot 3: Time vs delta_G
p3 = plot(df_log[:, :delta_G], df_log[:, :time],
          seriestype=:scatter, xscale=:log10,
          xlabel="delta_G", ylabel="Time (seconds)",
          title="Execution Time vs delta_G", legend=false)
plot!(p3, gdf[:, :delta_G], gdf[:, :mean_time],
      seriestype=:line, linewidth=2, color=:red, label="mean")

# Plot 4: Number of iterations vs delta_G
p4 = plot(df_log[:, :delta_G], df_log[:, :num_iters],
          seriestype=:scatter, xscale=:log10,
          xlabel="delta_G", ylabel="Number of Iterations",
          title="Iterations vs delta_G", legend=false)
plot!(p4, gdf[:, :delta_G], gdf[:, :mean_iters],
      seriestype=:line, linewidth=2, color=:red, label="mean")

# Combine plots
plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))
savefig(joinpath(@__DIR__, "results", "sweep_analysis_julia.pdf"))

# Additional grouped analysis by window type
windows = unique(df_log[:, :window])
p5 = plot(xscale=:log10)
for window in windows
    window_data = filter(:window => x -> x == window, df_log)
    plot!(p5, window_data[:, :delta_G], window_data[:, :average_cost],
          seriestype=:scatter, label=string(window), markershape=:circle)
end
# Overlay mean line across all data, grouped by delta_G
plot!(p5, gdf[:, :delta_G], gdf[:, :mean_cost],
      seriestype=:line, linewidth=2, color=:black, label="mean")
xlabel!("delta_G")
ylabel!("Average Cost")
title!("Average Cost vs delta_G by Window Type")
savefig(joinpath(@__DIR__, "results", "sweep_analysis_by_window.pdf"))

println("Plots saved to results/ directory")
