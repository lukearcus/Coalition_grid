# observability.jl
# Inverse optimization: given observed grid trades z*, find the set of private
# parameters (net_load, Q_b, P_b, SoC_0) consistent with z*.
#
# Variant A (Feasibility): pure LP — what net_load is consistent with z* and
#   *some* feasible battery operation? (Physical possibility.)
# Variant B (Optimality): LP with fixed duals from the forward solve — what
#   net_load is consistent with z* being *optimal*? (Rational adversary.)
#   Variant B is a local approximation: valid where the active set is unchanged.

include("Buildings.jl")
include("MPC_optimiser.jl")
include("load_EMS_data.jl")

using JuMP, SCS, HiGHS, LinearAlgebra, Printf, DataFrames, Dates, Plots

try; using CSV; catch; using CSV; end
@eval CSV.Parsers import Base.Ryu: writeshortest

# Physical bounds for inverse problem (adversary's prior knowledge)
const P_MAX   = 500.0    # max battery power (kW)
const Q_MAX   = 2000.0   # max battery capacity (kWh)
const NL_MAX  = 1000.0   # max |net_load| (kW)

# ===========================================================================
# Forward solve with clean dual extraction
# ===========================================================================

struct ForwardResult
    z_star::Vector{Float64}       # observed grid trade: g_buy - g_sell
    g_buy::Vector{Float64}
    g_sell::Vector{Float64}
    pos_δ::Vector{Float64}
    neg_δ::Vector{Float64}
    charge::Vector{Float64}
    # duals (JuMP convention for minimization)
    μ::Vector{Float64}            # power balance (equality, free)
    λ_init::Float64               # initial SoC (equality, free)
    λ_dyn::Vector{Float64}        # battery dynamics t=2..h (equality, free)
    σ_hi::Vector{Float64}         # charge <= Q_b (<=, dual >= 0)
    α_prime::Vector{Float64}      # pos_d <= P_b (<=, dual >= 0)
    β_prime::Vector{Float64}      # -neg_d <= P_b (<=, dual >= 0)
    φ::Float64                    # final constraint (>=, dual <= 0 in JuMP)
    # true parameters
    net_load_true::Vector{Float64}
    Q_b::Float64
    P_b::Float64
    SoC0::Float64
    # known parameters
    pb::Vector{Float64}
    ps::Vector{Float64}
    η_ch::Float64
    η_dis::Float64
    primal_obj::Float64
    dual_obj::Float64
    h::Int
end

function forward_solve(b::MPC_Building, k::Int, h::Int, opt::MPC_optimiser)
    η_ch = Float64(charge_eff(b))
    η_dis = Float64(discharge_eff(b))
    Q_b = Float64(max_store(b))
    P_b = Float64(max_flow(b))
    SoC0 = b.SoC[k]
    net_load_true = pred_consumption(b, k, h) .- pred_production(b, k, h)
    pb = energy_cost_k(opt, k, h)
    ps = energy_sale_k(opt, k, h)

    model = Model(SCS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "eps_abs", 1e-6)
    set_optimizer_attribute(model, "eps_rel", 1e-6)

    @variable(model, g_buy[1:h] >= 0)
    @variable(model, g_sell[1:h] >= 0)
    @variable(model, pos_δ[1:h] >= 0)
    @variable(model, neg_δ[1:h] <= 0)
    @variable(model, charge[1:h] >= 0)

    # Power balance: net_load + g_sell + pos_d + neg_d = g_buy
    @constraint(model, power_c[t=1:h],
        net_load_true[t] + g_sell[t] + pos_δ[t] + neg_δ[t] == g_buy[t])

    # Battery dynamics
    @constraint(model, init_c, charge[1] == SoC0)
    @constraint(model, dyn_c[t=2:h],
        charge[t] == charge[t-1] + η_ch * pos_δ[t-1] + (1/η_dis) * neg_δ[t-1])

    # Upper bound constraints (involve private params P_b, Q_b)
    @constraint(model, pos_cap[t=1:h], pos_δ[t] <= P_b)
    @constraint(model, neg_cap[t=1:h], -neg_δ[t] <= P_b)
    @constraint(model, charge_cap[t=1:h], charge[t] <= Q_b)

    # Final constraint
    @constraint(model, final_c, pos_δ[h] + neg_δ[h] + charge[h] >= 0)

    @objective(model, Min, sum(pb[t] * g_buy[t] - ps[t] * g_sell[t] for t in 1:h))

    optimize!(model)

    @assert termination_status(model) == MOI.OPTIMAL "Forward solve failed: $(termination_status(model))"

    z_star = value.(g_buy) .- value.(g_sell)

    # Extract duals
    μ = dual.(power_c)
    λ_init = dual(init_c)
    λ_dyn = [dual(dyn_c[t]) for t in 2:h]
    σ_hi = dual.(charge_cap)
    α_prime = dual.(pos_cap)
    β_prime = dual.(neg_cap)
    φ = dual(final_c)  # >= constraint: dual <= 0 in JuMP

    pobj = objective_value(model)
    dobj = dual_objective_value(model)

    return ForwardResult(z_star, value.(g_buy), value.(g_sell),
        value.(pos_δ), value.(neg_δ), value.(charge),
        μ, λ_init, λ_dyn, σ_hi, α_prime, β_prime, φ,
        net_load_true, Q_b, P_b, SoC0,
        pb, ps, η_ch, η_dis, pobj, dobj, h)
end

# ===========================================================================
# Inverse Feasibility LP (Variant A)
# ===========================================================================
# Pure LP: find (theta, x) such that x is primal-feasible with observed z*.
# No optimality condition — just physical possibility.

struct InverseBounds
    θ_min::Float64
    θ_max::Float64
    θ_true::Float64
    status_min::String
    status_max::String
end

function solve_inverse_feasibility(
        z_star::Vector{Float64}, h::Int, η_ch::Float64, η_dis::Float64,
        pb::Vector{Float64}, ps::Vector{Float64};
        objective_var::Symbol, objective_t::Int = 1, sense::Symbol = :min,
        known_battery::Bool = false,
        Q_b_val::Float64 = 0.0, P_b_val::Float64 = 0.0, SoC0_val::Float64 = 0.0)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "presolve", "on")

    # Private parameters (theta)
    @variable(model, net_load[1:h])
    if known_battery
        Q_b = Q_b_val
        P_b = P_b_val
        @variable(model, SoC_0 == SoC0_val)
    else
        @variable(model, 0 <= Q_b <= Q_MAX)
        @variable(model, 0 <= P_b <= P_MAX)
        @variable(model, 0 <= SoC_0)
    end

    # Unobserved primal
    @variable(model, g_buy[1:h] >= 0)
    @variable(model, g_sell[1:h] >= 0)
    @variable(model, pos_δ[1:h] >= 0)
    @variable(model, neg_δ[1:h] <= 0)
    @variable(model, charge[1:h] >= 0)

    # Observed grid trade
    @constraint(model, obs[t=1:h], g_buy[t] - g_sell[t] == z_star[t])

    # Power balance: net_load + g_sell + pos_d + neg_d = g_buy
    @constraint(model, pb_c[t=1:h],
        net_load[t] + g_sell[t] + pos_δ[t] + neg_δ[t] == g_buy[t])

    # Battery dynamics
    @constraint(model, init_c, charge[1] == SoC_0)
    @constraint(model, dyn_c[t=2:h],
        charge[t] == charge[t-1] + η_ch * pos_δ[t-1] + (1/η_dis) * neg_δ[t-1])

    # Capacity constraints (variable upper bounds -> linear constraints)
    if known_battery
        @constraint(model, pos_cap[t=1:h], pos_δ[t] <= P_b)
        @constraint(model, neg_cap[t=1:h], -neg_δ[t] <= P_b)
        @constraint(model, charge_cap[t=1:h], charge[t] <= Q_b)
        @constraint(model, soc_cap, SoC_0 <= Q_b)
    else
        @constraint(model, pos_cap[t=1:h], pos_δ[t] <= P_b)
        @constraint(model, neg_cap[t=1:h], -neg_δ[t] <= P_b)
        @constraint(model, charge_cap[t=1:h], charge[t] <= Q_b)
        @constraint(model, soc_cap, SoC_0 <= Q_b)
    end

    # Final constraint
    @constraint(model, final_c, pos_δ[h] + neg_δ[h] + charge[h] >= 0)

    # Physical bounds on net_load
    @constraint(model, nl_lo[t=1:h], net_load[t] >= -NL_MAX)
    @constraint(model, nl_hi[t=1:h], net_load[t] <= NL_MAX)

    # Objective: minimize or maximize the chosen parameter
    if objective_var == :net_load
        obj = net_load[objective_t]
    elseif objective_var == :Q_b
        known_battery && return Q_b, "FIXED"
        obj = Q_b
    elseif objective_var == :P_b
        known_battery && return P_b, "FIXED"
        obj = P_b
    elseif objective_var == :SoC_0
        known_battery && return SoC0_val, "FIXED"
        obj = SoC_0
    else
        error("Unknown objective_var: $objective_var")
    end

    if sense == :min
        @objective(model, Min, obj)
    else
        @objective(model, Max, obj)
    end

    optimize!(model)

    status = string(termination_status(model))
    val = (status == "OPTIMAL" || primal_status(model) == MOI.FEASIBLE_POINT) ? value(obj) : NaN
    return val, status
end

# ===========================================================================
# Inverse Optimality LP (Variant B — dual-fixed)
# ===========================================================================
# Adds strong duality (with fixed duals from forward solve) and complementary
# slackness (active constraints enforced as equalities).

function solve_inverse_optimality(
        fr::ForwardResult;
        objective_var::Symbol, objective_t::Int = 1, sense::Symbol = :min,
        tol::Float64 = 1e-4, known_battery::Bool = false)

    h = fr.h
    η_ch = fr.η_ch
    η_dis = fr.η_dis
    z_star = fr.z_star
    pb = fr.pb
    ps = fr.ps

    # Fixed duals from forward solve
    μ_star = fr.μ
    λ_init_star = fr.λ_init
    σ_hi_star = fr.σ_hi
    α_prime_star = fr.α_prime
    β_prime_star = fr.β_prime
    φ_star = fr.φ

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "presolve", "on")

    # Private parameters
    @variable(model, net_load[1:h])
    if known_battery
        Q_b_val = fr.Q_b
        P_b_val = fr.P_b
        @variable(model, SoC_0 == fr.SoC0)
    else
        @variable(model, 0 <= Q_b <= Q_MAX)
        @variable(model, 0 <= P_b <= P_MAX)
        @variable(model, 0 <= SoC_0)
    end

    # Unobserved primal
    @variable(model, g_buy[1:h] >= 0)
    @variable(model, g_sell[1:h] >= 0)
    @variable(model, pos_δ[1:h] >= 0)
    @variable(model, neg_δ[1:h] <= 0)
    @variable(model, charge[1:h] >= 0)

    # --- Primal feasibility + observed z ---
    @constraint(model, obs[t=1:h], g_buy[t] - g_sell[t] == z_star[t])
    @constraint(model, pb_c[t=1:h],
        net_load[t] + g_sell[t] + pos_δ[t] + neg_δ[t] == g_buy[t])
    @constraint(model, init_c, charge[1] == SoC_0)
    @constraint(model, dyn_c[t=2:h],
        charge[t] == charge[t-1] + η_ch * pos_δ[t-1] + (1/η_dis) * neg_δ[t-1])
    if known_battery
        @constraint(model, pos_cap[t=1:h], pos_δ[t] <= Q_b_val)
        @constraint(model, neg_cap[t=1:h], -neg_δ[t] <= P_b_val)
        @constraint(model, charge_cap[t=1:h], charge[t] <= Q_b_val)
    else
        @constraint(model, pos_cap[t=1:h], pos_δ[t] <= P_b)
        @constraint(model, neg_cap[t=1:h], -neg_δ[t] <= P_b)
        @constraint(model, charge_cap[t=1:h], charge[t] <= Q_b)
        @constraint(model, soc_cap, SoC_0 <= Q_b)
    end
    @constraint(model, final_c, pos_δ[h] + neg_δ[h] + charge[h] >= 0)
    @constraint(model, nl_lo[t=1:h], net_load[t] >= -NL_MAX)
    @constraint(model, nl_hi[t=1:h], net_load[t] <= NL_MAX)

    # --- Strong duality (with fixed duals, relaxed for SCS tolerance) ---
    # SCS duals at 1e-4 tolerance make exact equality infeasible; use a band.
    sd_tol = max(0.01, 1e-4 * abs(fr.primal_obj))
    if known_battery
        @constraint(model, strong_dual_lo,
            sum(pb[t] * g_buy[t] - ps[t] * g_sell[t] for t in 1:h)
            - (-sum(net_load[t] * μ_star[t] for t in 1:h)
               + SoC_0 * λ_init_star
               + Q_b_val * sum(σ_hi_star)
               + P_b_val * sum(α_prime_star)
               + P_b_val * sum(β_prime_star)) <= sd_tol)
        @constraint(model, strong_dual_hi,
            (-sum(net_load[t] * μ_star[t] for t in 1:h)
             + SoC_0 * λ_init_star
             + Q_b_val * sum(σ_hi_star)
             + P_b_val * sum(α_prime_star)
             + P_b_val * sum(β_prime_star))
            - sum(pb[t] * g_buy[t] - ps[t] * g_sell[t] for t in 1:h) <= sd_tol)
    else
        @constraint(model, strong_dual_lo,
            sum(pb[t] * g_buy[t] - ps[t] * g_sell[t] for t in 1:h)
            - (-sum(net_load[t] * μ_star[t] for t in 1:h)
               + SoC_0 * λ_init_star
               + Q_b * sum(σ_hi_star)
               + P_b * sum(α_prime_star)
               + P_b * sum(β_prime_star)) <= sd_tol)
        @constraint(model, strong_dual_hi,
            (-sum(net_load[t] * μ_star[t] for t in 1:h)
             + SoC_0 * λ_init_star
             + Q_b * sum(σ_hi_star)
             + P_b * sum(α_prime_star)
             + P_b * sum(β_prime_star))
            - sum(pb[t] * g_buy[t] - ps[t] * g_sell[t] for t in 1:h) <= sd_tol)
    end

    # --- Complementary slackness (active constraints with nonzero duals) ---
    for t in 1:h
        if abs(σ_hi_star[t]) > tol
            if known_battery
                @constraint(model, charge[t] == Q_b_val)
            else
                @constraint(model, charge[t] == Q_b)
            end
        end
        if abs(α_prime_star[t]) > tol
            if known_battery
                @constraint(model, pos_δ[t] == P_b_val)
            else
                @constraint(model, pos_δ[t] == P_b)
            end
        end
        if abs(β_prime_star[t]) > tol
            if known_battery
                @constraint(model, -neg_δ[t] == P_b_val)
            else
                @constraint(model, -neg_δ[t] == P_b)
            end
        end
    end
    if abs(φ_star) > tol
        @constraint(model, pos_δ[h] + neg_δ[h] + charge[h] == 0)
    end

    # Objective
    if objective_var == :net_load
        obj = net_load[objective_t]
    elseif objective_var == :Q_b
        known_battery && return fr.Q_b, "FIXED"
        obj = Q_b
    elseif objective_var == :P_b
        known_battery && return fr.P_b, "FIXED"
        obj = P_b
    elseif objective_var == :SoC_0
        known_battery && return fr.SoC0, "FIXED"
        obj = SoC_0
    else
        error("Unknown objective_var: $objective_var")
    end

    if sense == :min
        @objective(model, Min, obj)
    else
        @objective(model, Max, obj)
    end

    optimize!(model)

    status = string(termination_status(model))
    val = (status == "OPTIMAL" || primal_status(model) == MOI.FEASIBLE_POINT) ? value(obj) : NaN
    return val, status
end

# ===========================================================================
# Dual verification (sanity check)
# ===========================================================================

function verify_duals(fr::ForwardResult; tol = 1e-3)
    h = fr.h
    println("\n=== Dual Verification ===")
    println("Primal obj: $(fr.primal_obj)")
    println("Dual obj:   $(fr.dual_obj)")
    println("Match: $(abs(fr.primal_obj - fr.dual_obj) < tol)")

    # Check μ bounds: p_sell <= -μ <= p_buy (SCS negates equality duals)
    for t in 1:h
        neg_μ = -fr.μ[t]
        if neg_μ < fr.ps[t] - tol || neg_μ > fr.pb[t] + tol
            println("  WARN: -μ[$t]=$(neg_μ) outside [$(fr.ps[t]), $(fr.pb[t])]")
        end
    end

    # Check complementary slackness (|dual| > 0 => constraint active)
    for t in 1:h
        if abs(fr.σ_hi[t]) > tol && abs(fr.charge[t] - fr.Q_b) > tol
            println("  WARN: |σ_hi[$t]|=$(abs(fr.σ_hi[t])) but charge[$t]=$(fr.charge[t]) != Q_b=$(fr.Q_b)")
        end
        if abs(fr.α_prime[t]) > tol && abs(fr.pos_δ[t] - fr.P_b) > tol
            println("  WARN: |α'[$t]|=$(abs(fr.α_prime[t])) but pos_d[$t]=$(fr.pos_δ[t]) != P_b=$(fr.P_b)")
        end
        if abs(fr.β_prime[t]) > tol && abs(-fr.neg_δ[t] - fr.P_b) > tol
            println("  WARN: |β'[$t]|=$(abs(fr.β_prime[t])) but -neg_d[$t]=$(-fr.neg_δ[t]) != P_b=$(fr.P_b)")
        end
    end
    println("Verification complete.\n")
end

# ===========================================================================
# Helper: compute all bounds for a forward result
# ===========================================================================

function compute_all_bounds(fr::ForwardResult; known_battery::Bool = false, feas::Bool = true)
    h = fr.h
    results = DataFrame(
        param = String[], t = Int[],
        theta_min_feas = Float64[], theta_max_feas = Float64[],
        theta_min_opt = Float64[], theta_max_opt = Float64[],
        theta_true = Float64[],
        ratio_feas = Float64[], ratio_opt = Float64[],
    )

    kb_args = if known_battery
        (known_battery=true, Q_b_val=fr.Q_b, P_b_val=fr.P_b, SoC0_val=fr.SoC0)
    else
        (known_battery=false,)
    end

    for t in 1:h
        if feas
            v_min, _ = solve_inverse_feasibility(fr.z_star, h, fr.η_ch, fr.η_dis,
                fr.pb, fr.ps; objective_var=:net_load, objective_t=t, sense=:min, kb_args...)
            v_max, _ = solve_inverse_feasibility(fr.z_star, h, fr.η_ch, fr.η_dis,
                fr.pb, fr.ps; objective_var=:net_load, objective_t=t, sense=:max, kb_args...)
        else
            v_min, v_max = NaN, NaN
        end
        o_min, _ = solve_inverse_optimality(fr;
            objective_var=:net_load, objective_t=t, sense=:min, known_battery=known_battery)
        o_max, _ = solve_inverse_optimality(fr;
            objective_var=:net_load, objective_t=t, sense=:max, known_battery=known_battery)

        θ_true = fr.net_load_true[t]
        denom = max(abs(θ_true), 1.0)
        push!(results, ["net_load", t, v_min, v_max, o_min, o_max, θ_true,
            feas ? (v_max - v_min) / denom : NaN, (o_max - o_min) / denom])
    end

    if feas
        for param in (:Q_b, :P_b, :SoC_0)
            v_min, _ = solve_inverse_feasibility(fr.z_star, h, fr.η_ch, fr.η_dis,
                fr.pb, fr.ps; objective_var=param, sense=:min, kb_args...)
            v_max, _ = solve_inverse_feasibility(fr.z_star, h, fr.η_ch, fr.η_dis,
                fr.pb, fr.ps; objective_var=param, sense=:max, kb_args...)
            o_min, _ = solve_inverse_optimality(fr;
                objective_var=param, sense=:min, known_battery=known_battery)
            o_max, _ = solve_inverse_optimality(fr;
                objective_var=param, sense=:max, known_battery=known_battery)

            θ_true = param == :Q_b ? fr.Q_b : param == :P_b ? fr.P_b : fr.SoC0
            denom = max(abs(θ_true), 1.0)
            push!(results, [string(param), 0, v_min, v_max, o_min, o_max, θ_true,
                (v_max - v_min) / denom, (o_max - o_min) / denom])
        end
    end

    return results
end

function print_bounds_table(results::DataFrame, title::String)
    println("\n=== $title ===")
    println("=" ^ 120)
    @printf("%-12s %4s %12s %12s %12s %12s %12s %10s %10s\n",
        "Param", "t", "min_feas", "max_feas", "min_opt", "max_opt", "true", "ratio_feas", "ratio_opt")
    println("-" ^ 120)
    for row in eachrow(results)
        @printf("%-12s %4d %12.3f %12.3f %12.3f %12.3f %12.3f %10.3f %10.3f\n",
            row.param, row.t, row.theta_min_feas, row.theta_max_feas,
            row.theta_min_opt, row.theta_max_opt, row.theta_true,
            row.ratio_feas, row.ratio_opt)
    end
    println("=" ^ 120)
end

# ===========================================================================
# Main analysis
# ===========================================================================

function main()
    num_builds = 10
    num_steps = 96
    all_buildings, energy_cost, energy_sale = MPC_load_from_CSV(num_builds, num_steps)
    global opt = MPC_optimiser(energy_cost', energy_sale')

    k = 20  # MPC step (spans price change at t=24: 0.13 -> 0.17)
    h = 16  # horizon

    # -----------------------------------------------------------------------
    # Part 1: Building 1, unknown vs known battery
    # -----------------------------------------------------------------------
    println("\n" * "=" ^ 80)
    println("PART 1: Building 1 — Unknown vs Known Battery Parameters")
    println("=" ^ 80)

    b = all_buildings[1]
    fr = forward_solve(b, k, h, opt)
    println("Building 1: Q_b=$(fr.Q_b), P_b=$(fr.P_b), SoC_0=$(fr.SoC0)")
    println("z* = $(round.(fr.z_star, digits=3))")
    println("net_load_true = $(round.(fr.net_load_true, digits=3))")
    verify_duals(fr)

    res_unknown = compute_all_bounds(fr; known_battery=false, feas=true)
    print_bounds_table(res_unknown, "Building 1, h=$h, Battery UNKNOWN")

    res_known = compute_all_bounds(fr; known_battery=true, feas=false)
    print_bounds_table(res_known, "Building 1, h=$h, Battery KNOWN")

    # Plot: net_load bounds comparison
    nl_unknown = res_unknown[res_unknown.param .== "net_load", :]
    nl_known   = res_known[res_known.param .== "net_load", :]

    p1 = plot(1:h, nl_unknown.theta_true, label="true", lw=2, color=:black)
    plot!(1:h, nl_unknown.theta_min_opt, fillrange=nl_unknown.theta_max_opt,
        alpha=0.3, color=:red, label="opt bounds (unknown batt)")
    plot!(1:h, nl_unknown.theta_min_feas, fillrange=nl_unknown.theta_max_feas,
        alpha=0.15, color=:blue, label="feas bounds (unknown batt)")
    plot!(1:h, nl_known.theta_min_opt, fillrange=nl_known.theta_max_opt,
        alpha=0.3, color=:green, label="opt bounds (known batt)")
    xlabel!("timestep t"); ylabel!("net_load (kW)")
    title!("Observability of net_load (Building 1, h=$h)")
    savefig(p1, "results/observability_net_load_bldg1_h$h.pdf")
    println("Saved: results/observability_net_load_bldg1_h$h.pdf")

    # -----------------------------------------------------------------------
    # Part 2: Cross-building comparison
    # -----------------------------------------------------------------------
    println("\n" * "=" ^ 80)
    println("PART 2: Cross-Building Comparison")
    println("=" ^ 80)

    building_ids = [1, 3, 4]  # Q_b = 300, 50, 500; P_b = 75, 12.5, 125
    cross_results = DataFrame(
        building = Int[], Q_b = Float64[], P_b = Float64[],
        mean_ratio_feas = Float64[], mean_ratio_opt = Float64[],
        mean_ratio_opt_kb = Float64[],
    )

    for bid in building_ids
        println("  Building $bid...")
        b = all_buildings[bid]
        fr = forward_solve(b, k, h, opt)
        res = compute_all_bounds(fr; known_battery=false, feas=false)
        res_kb = compute_all_bounds(fr; known_battery=true, feas=false)

        nl_res = res[res.param .== "net_load", :]
        nl_res_kb = res_kb[res_kb.param .== "net_load", :]

        push!(cross_results, (bid, fr.Q_b, fr.P_b,
            NaN, mean(nl_res.ratio_opt), mean(nl_res_kb.ratio_opt)))

        println("Building $bid: Q_b=$(fr.Q_b), P_b=$(fr.P_b) | "
                * "opt=$(round(mean(nl_res.ratio_opt), digits=2)), "
                * "opt+kb=$(round(mean(nl_res_kb.ratio_opt), digits=2))")
    end

    # Plot: mean observability ratio by building
    p2 = plot(cross_results.building, cross_results.mean_ratio_opt,
        marker=:s, label="Optimality (unknown batt)", lw=2)
    plot!(cross_results.building, cross_results.mean_ratio_opt_kb,
        marker=:^, label="Optimality (known batt)", lw=2)
    xlabel!("Building ID"); ylabel!("Mean observability ratio")
    title!("net_load Observability by Building")
    savefig(p2, "results/observability_cross_building.pdf")
    println("Saved: results/observability_cross_building.pdf")

    # -----------------------------------------------------------------------
    # Part 3: Horizon sensitivity
    # -----------------------------------------------------------------------
    println("\n" * "=" ^ 80)
    println("PART 3: Horizon Sensitivity (Building 1)")
    println("=" ^ 80)

    horizons = [4, 8, 16, 32]
    b = all_buildings[1]
    hor_results = DataFrame(
        h = Int[], mean_ratio_opt = Float64[], mean_ratio_opt_kb = Float64[],
    )

    for h_val in horizons
        println("  h=$h_val...")
        k_val = max(1, k - (h - horizons[1]) ÷ 2)  # keep centered on price change
        k_val = min(k_val, 96 - h_val)
        fr_h = forward_solve(b, k_val, h_val, opt)
        res_h = compute_all_bounds(fr_h; known_battery=false, feas=false)
        res_h_kb = compute_all_bounds(fr_h; known_battery=true, feas=false)

        nl_h = res_h[res_h.param .== "net_load", :]
        nl_h_kb = res_h_kb[res_h_kb.param .== "net_load", :]

        push!(hor_results, (h_val, mean(nl_h.ratio_opt), mean(nl_h_kb.ratio_opt)))

        println("h=$h_val: opt=$(round(mean(nl_h.ratio_opt), digits=2)), "
                * "opt+kb=$(round(mean(nl_h_kb.ratio_opt), digits=2))")
    end

    p3 = plot(hor_results.h, hor_results.mean_ratio_opt,
        marker=:s, label="Optimality (unknown batt)", lw=2)
    plot!(hor_results.h, hor_results.mean_ratio_opt_kb,
        marker=:^, label="Optimality (known batt)", lw=2)
    xlabel!("Horizon length h"); ylabel!("Mean observability ratio")
    title!("net_load Observability vs Horizon (Building 1)")
    savefig(p3, "results/observability_horizon_sensitivity.pdf")
    println("Saved: results/observability_horizon_sensitivity.pdf")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    println("\n" * "=" ^ 80)
    println("SUMMARY")
    println("=" ^ 80)
    println("Cross-building:")
    println(cross_results)
    println("\nHorizon sensitivity:")
    println(hor_results)
    println("\nAll plots saved to results/observability_*.pdf")

    # Save results to CSV
    CSV.write("results/observability_bldg1_unknown.csv", res_unknown)
    CSV.write("results/observability_bldg1_known.csv", res_known)
    CSV.write("results/observability_cross_building.csv", cross_results)
    CSV.write("results/observability_horizon.csv", hor_results)
    println("Results saved to results/observability_*.csv")

    return res_unknown, res_known, cross_results, hor_results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
