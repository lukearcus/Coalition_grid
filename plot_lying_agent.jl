using DataFrames, Plots, Statistics
try
    using CSV
catch
    using CSV
end

df = CSV.read(joinpath(@__DIR__, "results", "sweep_lying_agent.csv"), DataFrame)
gr()

LIE_COLORS = Dict("truthful" => :blue, "bounded" => :orange, "fabrication" => :red)
LIE_LABELS = Dict("truthful" => "Truthful", "bounded" => "Bounded lie", "fabrication" => "Fabrication")

df_log = filter(:delta_G => x -> x > 0, df)

# --- Plot 1: honest_avg_cost vs delta_G, per lie_level ---
p1 = plot(xscale=:log10, xlabel=raw"delta_G", ylabel="Honest agents avg cost",
          title="Honest Agents' Cost vs delta_G", legend=:topright)

for ll in ["truthful", "bounded", "fabrication"]
    sub = filter(:lie_level => ==(ll), df_log)
    isempty(sub) && continue
    gdf = combine(groupby(sub, :delta_G),
                  :honest_avg_cost => mean => :mean_cost,
                  :honest_avg_cost => std  => :std_cost)
    sort!(gdf, :delta_G)
    xs = gdf[:, :delta_G]
    ms = gdf[:, :mean_cost]
    ss = gdf[:, :std_cost]
    lower = [isnothing(s) || isnan(s) ? m : m - s for (m, s) in zip(ms, ss)]
    upper = [isnothing(s) || isnan(s) ? m : m + s for (m, s) in zip(ms, ss)]
    plot!(p1, xs, ms, seriestype=:line, linewidth=2, color=LIE_COLORS[ll], label=LIE_LABELS[ll])
    plot!(p1, xs, lower, fillrange=upper, seriestype=:path, color=LIE_COLORS[ll], alpha=0.15, lw=0, label=nothing)
    scatter!(p1, sub[:, :delta_G], sub[:, :honest_avg_cost], color=LIE_COLORS[ll], alpha=0.2, ms=2, label=nothing)
end

# delta_G=0 anchor markers
df0 = filter(:delta_G => x -> x == 0.0, df)
if !isempty(df0)
    x_anchor = 10^(log10(minimum(df_log[:, :delta_G])) - 0.5)
    for ll in ["truthful", "bounded", "fabrication"]
        sub0 = filter(:lie_level => ==(ll), df0)
        isempty(sub0) && continue
        m0 = mean(sub0[:, :honest_avg_cost])
        scatter!(p1, [x_anchor], [m0], markershape=:star5, ms=8, color=LIE_COLORS[ll], label=LIE_LABELS[ll] * " (dg=0)")
    end
end

savefig(p1, joinpath(@__DIR__, "results", "lying_agent_costs.pdf"))
println("Saved results/lying_agent_costs.pdf")

# --- Plot 2: degradation vs delta_G ---
# Degradation = honest_avg_cost[lie] - honest_avg_cost[truthful], paired by (window, delta_G, repeat)
truthful = filter(:lie_level => ==("truthful"), df)
truthful = select(truthful, [:window, :delta_G, :repeat, :honest_avg_cost => :truthful_cost])

p2 = plot(xscale=:log10, xlabel=raw"delta_G", ylabel="Cost degradation (honest agents)",
          title="Degradation vs delta_G", legend=:topright)

for ll in ["bounded", "fabrication"]
    ll_df = filter(:lie_level => ==(ll), df)
    isempty(ll_df) && continue
    ll_sel = select(ll_df, [:window, :delta_G, :repeat, :honest_avg_cost => :lie_cost])
    deg = innerjoin(truthful, ll_sel, on=[:window, :delta_G, :repeat])
    deg.degradation = deg.lie_cost .- deg.truthful_cost

    deg_log = filter(:delta_G => x -> x > 0, deg)
    isempty(deg_log) && continue
    gdf = combine(groupby(deg_log, :delta_G),
                  :degradation => mean => :mean_deg,
                  :degradation => std  => :std_deg)
    sort!(gdf, :delta_G)
    xs = gdf[:, :delta_G]
    ms = gdf[:, :mean_deg]
    ss = gdf[:, :std_deg]
    lower = [isnothing(s) || isnan(s) ? m : m - s for (m, s) in zip(ms, ss)]
    upper = [isnothing(s) || isnan(s) ? m : m + s for (m, s) in zip(ms, ss)]
    plot!(p2, xs, ms, seriestype=:line, linewidth=2, color=LIE_COLORS[ll], label=LIE_LABELS[ll])
    plot!(p2, xs, lower, fillrange=upper, seriestype=:path, color=LIE_COLORS[ll], alpha=0.15, lw=0, label=nothing)
    scatter!(p2, deg_log[:, :delta_G], deg_log[:, :degradation], color=LIE_COLORS[ll], alpha=0.2, ms=2, label=nothing)
end

# Zero baseline
if !isempty(df_log)
    xmin = minimum(df_log[:, :delta_G])
    xmax = maximum(df_log[:, :delta_G])
    plot!(p2, [xmin, xmax], [0, 0], seriestype=:line, color=:black, linestyle=:dash, linewidth=1, label=nothing)
end

savefig(p2, joinpath(@__DIR__, "results", "lying_agent_degradation.pdf"))
println("Saved results/lying_agent_degradation.pdf")

# --- Plot 3: liar's own cost (for context) ---
p3 = plot(xscale=:log10, xlabel=raw"delta_G", ylabel="Liar's cost",
          title="Liar's Own Cost vs delta_G", legend=:topright)

for ll in ["truthful", "bounded", "fabrication"]
    sub = filter(:lie_level => ==(ll), df_log)
    isempty(sub) && continue
    gdf = combine(groupby(sub, :delta_G),
                  :liar_cost => mean => :mean_cost,
                  :liar_cost => std  => :std_cost)
    sort!(gdf, :delta_G)
    xs = gdf[:, :delta_G]
    ms = gdf[:, :mean_cost]
    ss = gdf[:, :std_cost]
    lower = [isnothing(s) || isnan(s) ? m : m - s for (m, s) in zip(ms, ss)]
    upper = [isnothing(s) || isnan(s) ? m : m + s for (m, s) in zip(ms, ss)]
    plot!(p3, xs, ms, seriestype=:line, linewidth=2, color=LIE_COLORS[ll], label=LIE_LABELS[ll])
    plot!(p3, xs, lower, fillrange=upper, seriestype=:path, color=LIE_COLORS[ll], alpha=0.15, lw=0, label=nothing)
    scatter!(p3, sub[:, :delta_G], sub[:, :liar_cost], color=LIE_COLORS[ll], alpha=0.2, ms=2, label=nothing)
end

if !isempty(df0)
    x_anchor = 10^(log10(minimum(df_log[:, :delta_G])) - 0.5)
    for ll in ["truthful", "bounded", "fabrication"]
        sub0 = filter(:lie_level => ==(ll), df0)
        isempty(sub0) && continue
        m0 = mean(sub0[:, :liar_cost])
        scatter!(p3, [x_anchor], [m0], markershape=:star5, ms=8, color=LIE_COLORS[ll], label=LIE_LABELS[ll] * " (dg=0)")
    end
end

savefig(p3, joinpath(@__DIR__, "results", "lying_agent_liar_cost.pdf"))
println("Saved results/lying_agent_liar_cost.pdf")

# --- Combined plot ---
plot(p1, p2, p3, layout=(1, 3), size=(1800, 500))
savefig(joinpath(@__DIR__, "results", "lying_agent_analysis.pdf"))
println("Saved results/lying_agent_analysis.pdf")
