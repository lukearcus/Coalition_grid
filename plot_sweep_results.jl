using DataFrames, Plots, Statistics
try
    using CSV
catch
    using CSV
end

# Read the CSV file
df = CSV.read(joinpath(@__DIR__, "results", "sweep_delta_G_subset.csv"), DataFrame)

# Set plotting backend
gr()

# Filter out delta_G == 0 (invalid for log scale, kept as a separate anchor)
df_log = filter(:delta_G => x -> x > 0, df)

# delta_G = 0 deterministic anchor: mean across its (identical) repeats
df0   = filter(:delta_G => x -> x == 0.0, df)
dg0   = combine(df0,
               :average_cost   => mean => :mean_cost,
               :average_cost   => std  => :std_cost,
               :time           => mean => :mean_time,
               :num_iters      => mean => :mean_iters)
# Cosmetic x-position for the anchor: half a decade below the smallest nonzero delta_G
x_anchor = 10^(log10(minimum(df_log[:, :delta_G])) - 0.5)

# Compute mean and std of each metric per delta_G value (across repeats), delta_G > 0
gdf = combine(groupby(df_log, :delta_G),
              :average_cost  => mean => :mean_cost,
              :average_cost  => std  => :std_cost,
              :time          => mean => :mean_time,
              :time          => std  => :std_time,
              :num_iters     => mean => :mean_iters,
              :num_iters     => std  => :std_iters)
sort!(gdf, :delta_G)

# Linear least-squares fit of y vs log10(x): returns (intercept, slope)
function loglinear_fit(xs, ys)
    X = hcat(ones(length(xs)), log10.(xs))
    coef = X \ ys
    return coef[1], coef[2]
end

# Shared styling: scatter raw, mean line, ±1 std ribbon, linear trend (in log-x),
# plus a delta_G=0 deterministic anchor at a cosmetic x-position.
function sweep_plot(df_log, gdf, dg0, x_anchor, col, sfx, ylabel, title)
    p = plot(df_log[:, :delta_G], df_log[:, col],
             seriestype=:scatter, xscale=:log10,
             xlabel=raw"epsilon ⋅ delta_G", ylabel=ylabel,
             title=title, legend=:topright, label="raw",
             markershape=:circle, markeralpha=0.35, ms=3)
    mean_sym = Symbol("mean_", sfx)
    std_sym  = Symbol("std_",  sfx)
    xs = gdf[:, :delta_G]
    ms = gdf[:, mean_sym]
    ss = gdf[:, std_sym]
    plot!(p, xs, ms, seriestype=:line, linewidth=2, color=:red, label="mean")
    # ±1 std ribbon around the mean (NaN-safe if single repeat)
    lower = [isnothing(s) || isnan(s) ? m : m - s for (m, s) in zip(ms, ss)]
    upper = [isnothing(s) || isnan(s) ? m : m + s for (m, s) in zip(ms, ss)]
    plot!(p, xs, lower, fillrange=upper,
          seriestype=:path, color=:red, alpha=0.15, lw=0, label="±1 std")
    # Linear trend line fitted to the means in log10(delta_G) space (excludes anchor)
    a, b = loglinear_fit(xs, ms)
    trend_x = range(minimum(xs), maximum(xs); length=100)
    plot!(p, trend_x, a .+ b .* log10.(trend_x),
          seriestype=:line, linewidth=2, color=:green, linestyle=:dot,
          label="trend (log-x fit)")
    # delta_G = 0 deterministic anchor (excluded from the trend fit)
    plot!(p, [x_anchor], [dg0[1, mean_sym]],
          seriestype=:scatter, markershape=:star5, ms=8, color=:black,
          label="δ_G=0 (det)")
    return p
end

p1 = sweep_plot(df_log, gdf, dg0, x_anchor, :average_cost, "cost",  "Average Cost",         "Average Cost vs delta_G")
# Plot 2: std of cost across repeats at each delta_G, with log-x linear trend + anchor (std=0 by construction)
xs_cost  = gdf[:, :delta_G]
std_cost = gdf[:, :std_cost]
p2 = plot(xs_cost, std_cost,
          seriestype=:scatter, xscale=:log10,
          xlabel="epsilon ⋅ delta_G", ylabel="Std (Average Cost)",
          title="Std of Cost vs delta_G", legend=:topright,
          label="std", markershape=:circle, ms=4, color=:red)
a2, b2 = loglinear_fit(xs_cost, std_cost)
trend_x2 = range(minimum(xs_cost), maximum(xs_cost); length=100)
plot!(p2, trend_x2, a2 .+ b2 .* log10.(trend_x2),
      seriestype=:line, linewidth=2, color=:green, linestyle=:dot,
      label="trend (log-x fit)")
plot!(p2, [x_anchor], [dg0[1, :std_cost]],
      seriestype=:scatter, markershape=:star5, ms=8, color=:black,
      label="δ_G=0 (det)")
p3 = sweep_plot(df_log, gdf, dg0, x_anchor, :time,      "time",  "Time (seconds)",       "Execution Time vs delta_G")
p4 = sweep_plot(df_log, gdf, dg0, x_anchor, :num_iters, "iters", "Number of Iterations", "Iterations vs delta_G")

# Combine plots
plot(p1, p2, p3, p4, layout=(2, 2), size=(1100, 850))
savefig(joinpath(@__DIR__, "results", "sweep_analysis_julia.pdf"))

println("Plots saved to results/ directory")
