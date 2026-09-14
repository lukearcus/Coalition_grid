# observability_coalition.jl
# Inverse optimization for coalition formation via ADMM.
#
# Three observation models:
#   Model A: Converged solution, observe z_i* only (net grid trade)
#   Model B: Converged solution, observe z_i* + coal_exch_i*
#   Model C: ADMM trajectory, observe proposed_coal/new_coal at each iteration
#
# The ADMM subproblem is a QP, but its KKT conditions are linear in the private
# parameters because coal_exch is observed (the quadratic term's gradient is
# c*(coal_exch - ct), which is a known constant when coal_exch is observed).
#
# At convergence (coal_exch = ct), the quadratic penalty vanishes and the KKT
# conditions reduce to those of the centralized coalition LP.

include("Buildings.jl")
include("MPC_optimiser.jl")
include("load_EMS_data.jl")

using JuMP, SCS, HiGHS, LinearAlgebra, Printf, DataFrames, Dates, Plots

try; using CSV; catch; using CSV; end
@eval CSV.Parsers import Base.Ryu: writeshortest

const P_MAX  = 500.0
const Q_MAX  = 2000.0
const NL_MAX = 1000.0

# ===========================================================================
# Trajectory recording
# ===========================================================================

struct ADMMTrajectory
    proposed_coals_hist::Vector{Matrix{Float64}}  # proposed_coals[j] = h x n_agents
    new_coals_hist::Vector{Matrix{Float64}}        # new_coals[j] = h x n_agents
    lambdas_hist::Vector{Matrix{Float64}}          # lambdas[j] = h x n_agents
    z_hist::Vector{Matrix{Float64}}               # z[j] = h x n_agents (per-iteration grid trade)
    # Per-iteration, per-agent duals extracted from the ADMM subproblem
    mu_hist::Vector{Matrix{Float64}}              # power balance dual: h x n_agents
    sigma_hist::Vector{Matrix{Float64}}            # charge <= Q_b dual: h x n_agents
    alpha_hist::Vector{Matrix{Float64}}            # pos_d <= P_b dual: h x n_agents
    beta_hist::Vector{Matrix{Float64}}             # -neg_d <= P_b dual: h x n_agents
    lambda_init_hist::Matrix{Float64}             # init SoC dual: n_iters x n_agents
    z_star::Matrix{Float64}                       # h x n_agents (converged net grid trade)
    coal_exch::Matrix{Float64}                    # h x n_agents (converged coalition exchange)
    n_iters::Int
    h::Int
    n_agents::Int
end

function run_admm_with_trajectory(bs::Vector{MPC_Building}, k::Int, h::Int, opt::MPC_optimiser; abs_tol::Float64=1e-5)
    n_agents = length(bs)
    num_steps = h
    c = 0.5
    new_coal = zeros(num_steps, n_agents)
    proposed_coal = [zeros(num_steps) for _ in 1:n_agents]
    lambdas = [zeros(num_steps) for _ in 1:n_agents]
    num_iters = 1
    max_admm_iters = 1000

    build_caches = [ADMM_build_opt_init(b, k, num_steps) for b in bs]
    coal_cache = ADMM_coal_update_init(n_agents, num_steps)

    proposed_coals_hist = Matrix{Float64}[]
    new_coals_hist = Matrix{Float64}[]
    lambdas_hist = Matrix{Float64}[]
    z_hist = Matrix{Float64}[]
    mu_hist = Matrix{Float64}[]
    sigma_hist = Matrix{Float64}[]
    alpha_hist = Matrix{Float64}[]
    beta_hist = Matrix{Float64}[]
    lambda_init_hist = Float64[]

    states = Vector{Any}(undef, n_agents)
    while (norm(reduce(hcat, proposed_coal) .- new_coal) > abs_tol) | (num_iters <= 1)
        if num_iters > max_admm_iters
            break
        end
        states = Vector{Any}(undef, n_agents)
        for ind in 1:n_agents
            states[ind] = ADMM_build_opt_solve(build_caches[ind], new_coal[:,ind], lambdas[ind], c)
        end
        proposed_coal = [value(state[6]) for state in states]

        new_coal = ADMM_coal_update_solve(coal_cache, lambdas, proposed_coal, c)

        lambdas = [lambda + c*(prop_coal - new_coal[:,ind])
                   for (lambda, prop_coal, ind) in zip(lambdas, proposed_coal, 1:n_agents)]

        push!(proposed_coals_hist, reduce(hcat, proposed_coal))
        push!(new_coals_hist, copy(new_coal))
        push!(lambdas_hist, reduce(hcat, lambdas))
        z_iter = zeros(num_steps, n_agents)
        mu_iter = zeros(num_steps, n_agents)
        sigma_iter = zeros(num_steps, n_agents)
        alpha_iter = zeros(num_steps, n_agents)
        beta_iter = zeros(num_steps, n_agents)
        for ind in 1:n_agents
            z_iter[:,ind] = value.(states[ind][2]) .- value.(states[ind][3])
            m = build_caches[ind].model
            mu_iter[:,ind] = dual.(m[:power_c])
            for t in 1:num_steps
                sigma_iter[t,ind] = dual(UpperBoundRef(m[:charge][t]))
                alpha_iter[t,ind] = dual(UpperBoundRef(m[:pos_delta_s][t]))
                beta_iter[t,ind] = dual(LowerBoundRef(m[:neg_delta_s][t]))
            end
            lambda_init_val = dual.(m[:charge_c])[1]
            push!(lambda_init_hist, lambda_init_val)
        end
        push!(z_hist, z_iter)
        push!(mu_hist, mu_iter)
        push!(sigma_hist, sigma_iter)
        push!(alpha_hist, alpha_iter)
        push!(beta_hist, beta_iter)
        num_iters += 1
    end

    # Extract converged solution from the ADMM (replacing coal_exch with consensus,
    # matching the original code at MPC_optimiser.jl:388-389)
    poss_sol = isempty(new_coals_hist) ? new_coal : new_coals_hist[end]
    z_star = zeros(num_steps, n_agents)
    coal_exch = zeros(num_steps, n_agents)
    for (ind, state) in enumerate(states)
        z_star[:,ind] = value.(state[2]) .- value.(state[3])
        coal_exch[:,ind] = poss_sol[:,ind]
    end

    n_iters_actual = num_iters - 1
    lambda_init_mat = reshape(lambda_init_hist, n_iters_actual, n_agents)

    return ADMMTrajectory(proposed_coals_hist, new_coals_hist, lambdas_hist, z_hist,
                          mu_hist, sigma_hist, alpha_hist, beta_hist, lambda_init_mat,
                          z_star, coal_exch, n_iters_actual, num_steps, n_agents)
end

# ===========================================================================
# Centralized coalition forward solve with dual extraction
# ===========================================================================

struct CoalForwardResult
    z_star::Matrix{Float64}        # h x n_agents
    coal_exch::Matrix{Float64}     # h x n_agents
    # Per-agent primal
    g_buy::Matrix{Float64}         # h x n_agents
    g_sell::Matrix{Float64}        # h x n_agents
    pos_d::Matrix{Float64}         # h x n_agents
    neg_d::Matrix{Float64}         # h x n_agents
    charge::Matrix{Float64}       # h x n_agents
    # Duals (from centralized LP)
    mu::Matrix{Float64}            # h x n_agents (power balance)
    coal_dual::Vector{Float64}     # h (coalition balance constraint)
    sigma_hi::Matrix{Float64}     # h x n_agents (charge <= Q_b)
    alpha_p::Matrix{Float64}       # h x n_agents (pos_d <= P_b)
    beta_p::Matrix{Float64}        # h x n_agents (-neg_d <= P_b)
    # True params per agent
    net_load_true::Matrix{Float64} # h x n_agents
    Q_b::Vector{Float64}           # n_agents
    P_b::Vector{Float64}           # n_agents
    SoC0::Vector{Float64}          # n_agents
    # Known params
    pb::Vector{Float64}
    ps::Vector{Float64}
    eta_ch::Float64
    eta_dis::Float64
    primal_obj::Float64
    h::Int
    n_agents::Int
end

function coal_forward_solve(bs::Vector{MPC_Building}, k::Int, h::Int, opt::MPC_optimiser)
    n_agents = length(bs)
    eta_ch = Float64(charge_eff(bs[1]))
    eta_dis = Float64(discharge_eff(bs[1]))
    Q_b = [Float64(max_store(b)) for b in bs]
    P_b = [Float64(max_flow(b)) for b in bs]
    SoC0 = [b.SoC[k] for b in bs]
    net_load_true = reduce(hcat, [pred_consumption(b,k,h) .- pred_production(b,k,h) for b in bs])
    pb = energy_cost_k(opt, k, h)
    ps = energy_sale_k(opt, k, h)

    model = Model(SCS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "eps_abs", 1e-6)
    set_optimizer_attribute(model, "eps_rel", 1e-6)

    @variable(model, g_buy[1:h, 1:n_agents] >= 0)
    @variable(model, g_sell[1:h, 1:n_agents] >= 0)
    @variable(model, pos_d[1:h, 1:n_agents] >= 0)
    @variable(model, neg_d[1:h, 1:n_agents] <= 0)
    @variable(model, charge[1:h, 1:n_agents] >= 0)
    @variable(model, coal_exch[1:h, 1:n_agents])

    # Power balance: net_load + g_sell + pos_d + neg_d + coal_exch = g_buy
    @constraint(model, power_c[t=1:h, i=1:n_agents],
        net_load_true[t,i] + g_sell[t,i] + pos_d[t,i] + neg_d[t,i] + coal_exch[t,i] == g_buy[t,i])

    # Battery dynamics
    @constraint(model, init_c[i=1:n_agents], charge[1,i] == SoC0[i])
    @constraint(model, dyn_c[t=2:h, i=1:n_agents],
        charge[t,i] == charge[t-1,i] + eta_ch * pos_d[t-1,i] + (1/eta_dis) * neg_d[t-1,i])

    # Capacity
    @constraint(model, pos_cap[t=1:h, i=1:n_agents], pos_d[t,i] <= P_b[i])
    @constraint(model, neg_cap[t=1:h, i=1:n_agents], -neg_d[t,i] <= P_b[i])
    @constraint(model, charge_cap[t=1:h, i=1:n_agents], charge[t,i] <= Q_b[i])

    # Coalition balance: sum of coal_exch = 0
    @constraint(model, coal_c[t=1:h], sum(coal_exch[t,:]) == 0)

    # Final constraint
    @constraint(model, final_c[i=1:n_agents], pos_d[h,i] + neg_d[h,i] + charge[h,i] >= 0)

    @objective(model, Min, sum(pb[t] * g_buy[t,i] - ps[t] * g_sell[t,i]
                               for t in 1:h, i in 1:n_agents))

    optimize!(model)
    @assert termination_status(model) == MOI.OPTIMAL "Coal forward solve failed"

    z_star = value.(g_buy) .- value.(g_sell)
    mu = dual.(power_c)
    coal_dual = dual.(coal_c)
    sigma_hi = dual.(charge_cap)
    alpha_p = dual.(pos_cap)
    beta_p = dual.(neg_cap)

    return CoalForwardResult(z_star, value.(coal_exch),
        value.(g_buy), value.(g_sell), value.(pos_d), value.(neg_d), value.(charge),
        mu, coal_dual, sigma_hi, alpha_p, beta_p,
        net_load_true, Q_b, P_b, SoC0, pb, ps, eta_ch, eta_dis,
        objective_value(model), h, n_agents)
end

# ===========================================================================
# Inverse LP: converged solution (Model A & B)
# ===========================================================================
# Model A: observe z_i* only. Power balance: net_load = z* - delta_s
# Model B: observe z_i* + coal_exch_i*. Power balance: net_load = (z* - coal_exch*) - delta_s
# Both use strong duality with duals from the centralized LP.

function solve_coal_inverse(fr::CoalForwardResult, agent::Int;
        objective_var::Symbol, objective_t::Int = 1, sense::Symbol = :min,
        model::Symbol = :B,  # :A or :B
        known_battery::Bool = false, tol::Float64 = 1e-4)

    h = fr.h
    eta_ch = fr.eta_ch
    eta_dis = fr.eta_dis
    pb = fr.pb
    ps = fr.ps
    Q_true = fr.Q_b[agent]
    P_true = fr.P_b[agent]
    SoC0_true = fr.SoC0[agent]

    # Observed quantities
    z_obs = fr.z_star[:,agent]
    ce_obs = model == :B ? fr.coal_exch[:,agent] : zeros(h)

    # Effective observation: net_load + delta_s = z* - coal_exch*
    eff_obs = z_obs .- ce_obs

    # Duals from centralized LP
    mu_star = fr.mu[:,agent]
    coal_dual_star = fr.coal_dual  # coalition balance dual (same for all agents)
    sigma_star = fr.sigma_hi[:,agent]
    alpha_star = fr.alpha_p[:,agent]
    beta_star = fr.beta_p[:,agent]

    m = Model(HiGHS.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "presolve", "on")

    # Private parameters
    @variable(m, net_load[1:h])
    if known_battery
        Q_b = Q_true
        P_b = P_true
        @variable(m, SoC_0 == SoC0_true)
    else
        @variable(m, 0 <= Q_b <= Q_MAX)
        @variable(m, 0 <= P_b <= P_MAX)
        @variable(m, 0 <= SoC_0)
    end

    # Unobserved primal
    @variable(m, g_buy[1:h] >= 0)
    @variable(m, g_sell[1:h] >= 0)
    @variable(m, pos_d[1:h] >= 0)
    @variable(m, neg_d[1:h] <= 0)
    @variable(m, charge[1:h] >= 0)
    @variable(m, coal_exch[1:h])  # unobserved in Model A, observed in B

    # Primal feasibility + observed trade
    @constraint(m, obs[t=1:h], g_buy[t] - g_sell[t] == z_obs[t])
    if model == :B
        @constraint(m, ce_obs_c[t=1:h], coal_exch[t] == ce_obs[t])
    end
    @constraint(m, pb_c[t=1:h],
        net_load[t] + g_sell[t] + pos_d[t] + neg_d[t] + coal_exch[t] == g_buy[t])
    @constraint(m, init_c, charge[1] == SoC_0)
    @constraint(m, dyn_c[t=2:h],
        charge[t] == charge[t-1] + eta_ch * pos_d[t-1] + (1/eta_dis) * neg_d[t-1])
    if known_battery
        @constraint(m, pos_cap[t=1:h], pos_d[t] <= P_b)
        @constraint(m, neg_cap[t=1:h], -neg_d[t] <= P_b)
        @constraint(m, charge_cap[t=1:h], charge[t] <= Q_b)
    else
        @constraint(m, pos_cap[t=1:h], pos_d[t] <= P_b)
        @constraint(m, neg_cap[t=1:h], -neg_d[t] <= P_b)
        @constraint(m, charge_cap[t=1:h], charge[t] <= Q_b)
        @constraint(m, soc_cap, SoC_0 <= Q_b)
    end
    @constraint(m, final_c, pos_d[h] + neg_d[h] + charge[h] >= 0)
    @constraint(m, nl_lo[t=1:h], net_load[t] >= -NL_MAX)
    @constraint(m, nl_hi[t=1:h], net_load[t] <= NL_MAX)

    # Strong duality (with fixed duals from centralized LP)
    # The centralized LP's strong duality involves ALL agents:
    #   sum_i sum_t (pb*gb_i - ps*gs_i) = -sum_i sum_t (nl_i*mu_i) + sum_i (Q_bi*sum(sig_i) + P_bi*sum(a_i) + P_bi*sum(b_i))
    # We model ALL agents' private params (net_load_i, Q_bi, P_bi, SoC0_i) and
    # all agents' unobserved primal, then optimize the target agent's parameter.
    sd_tol = max(0.01, 1e-4 * abs(fr.primal_obj))

    # Create per-agent variables for all OTHER agents
    other_nl = Dict{Int, Vector{VariableRef}}()
    other_Q = Dict{Int, Any}()
    other_P = Dict{Int, Any}()
    other_soc0 = Dict{Int, Any}()
    other_gb = Dict{Int, Vector{VariableRef}}()
    other_gs = Dict{Int, Vector{VariableRef}}()
    other_pd = Dict{Int, Vector{VariableRef}}()
    other_nd = Dict{Int, Vector{VariableRef}}()
    other_ch = Dict{Int, Vector{VariableRef}}()
    other_ce = Dict{Int, Vector{VariableRef}}()

    for i in 1:fr.n_agents
        i == agent && continue
        other_nl[i] = @variable(m, [1:h], base_name="nl_$i")
        @constraint(m, [t=1:h], other_nl[i][t] >= -NL_MAX)
        @constraint(m, [t=1:h], other_nl[i][t] <= NL_MAX)
        if known_battery
            other_Q[i] = fr.Q_b[i]
            other_P[i] = fr.P_b[i]
            @variable(m, SoC_0_i == fr.SoC0[i], base_name="soc0_$i")
            other_soc0[i] = SoC_0_i
        else
            @variable(m, 0 <= Q_b_i <= Q_MAX, base_name="Q_$i")
            @variable(m, 0 <= P_b_i <= P_MAX, base_name="P_$i")
            @variable(m, 0 <= SoC_0_i, base_name="soc0_$i")
            other_Q[i] = Q_b_i
            other_P[i] = P_b_i
            other_soc0[i] = SoC_0_i
        end
        other_gb[i] = @variable(m, [1:h], lower_bound=0, base_name="gb_$i")
        other_gs[i] = @variable(m, [1:h], lower_bound=0, base_name="gs_$i")
        other_pd[i] = @variable(m, [1:h], lower_bound=0, base_name="pd_$i")
        other_nd[i] = @variable(m, [1:h], upper_bound=0, base_name="nd_$i")
        other_ch[i] = @variable(m, [1:h], lower_bound=0, base_name="ch_$i")
        other_ce[i] = @variable(m, [1:h], base_name="ce_$i")

        ce_i_obs = model == :B ? fr.coal_exch[:,i] : nothing
        for t in 1:h
            @constraint(m, other_gb[i][t] - other_gs[i][t] == fr.z_star[t,i])
            if model == :B
                @constraint(m, other_ce[i][t] == ce_i_obs[t])
            end
            @constraint(m, other_nl[i][t] + other_gs[i][t] + other_pd[i][t] + other_nd[i][t] + other_ce[i][t] == other_gb[i][t])
            @constraint(m, other_pd[i][t] <= other_P[i])
            @constraint(m, -other_nd[i][t] <= other_P[i])
            @constraint(m, other_ch[i][t] <= other_Q[i])
        end
        @constraint(m, other_ch[i][1] == other_soc0[i])
        for t in 2:h
            @constraint(m, other_ch[i][t] == other_ch[i][t-1] + eta_ch * other_pd[i][t-1] + (1/eta_dis) * other_nd[i][t-1])
        end
        @constraint(m, other_pd[i][h] + other_nd[i][h] + other_ch[i][h] >= 0)
    end

    # Build the full strong duality constraint
    primal_all = sum(pb[t]*g_buy[t] - ps[t]*g_sell[t] for t in 1:h)
    dual_all = -sum(net_load[t]*mu_star[t] for t in 1:h) + Q_b*sum(sigma_star) + P_b*sum(alpha_star) + P_b*sum(beta_star)
    for i in 1:fr.n_agents
        i == agent && continue
        primal_all += sum(pb[t]*other_gb[i][t] - ps[t]*other_gs[i][t] for t in 1:h)
        dual_all += -sum(other_nl[i][t]*fr.mu[:,i][t] for t in 1:h) + other_Q[i]*sum(fr.sigma_hi[:,i]) + other_P[i]*sum(fr.alpha_p[:,i]) + other_P[i]*sum(fr.beta_p[:,i])
    end

    @constraint(m, primal_all - dual_all <= sd_tol)
    @constraint(m, dual_all - primal_all <= sd_tol)

    # Complementary slackness (for target agent only)
    for t in 1:h
        if abs(sigma_star[t]) > tol
            @constraint(m, charge[t] == Q_b)
        end
        if abs(alpha_star[t]) > tol
            @constraint(m, pos_d[t] == P_b)
        end
        if abs(beta_star[t]) > tol
            @constraint(m, -neg_d[t] == P_b)
        end
    end

    # Objective
    if objective_var == :net_load
        obj = net_load[objective_t]
    elseif objective_var == :Q_b
        known_battery && return Q_true, "FIXED"
        obj = Q_b
    elseif objective_var == :P_b
        known_battery && return P_true, "FIXED"
        obj = P_b
    elseif objective_var == :SoC_0
        known_battery && return SoC0_true, "FIXED"
        obj = SoC_0
    else
        error("Unknown objective_var: $objective_var")
    end

    sense == :min ? @objective(m, Min, obj) : @objective(m, Max, obj)
    optimize!(m)

    status = string(termination_status(m))
    val = (status == "OPTIMAL" || primal_status(m) == MOI.FEASIBLE_POINT) ? value(obj) : NaN
    return val, status
end

# ===========================================================================
# Inverse LP: trajectory-based (Model C)
# ===========================================================================
# One large LP with shared private params and per-iteration KKT conditions.
# Each ADMM iteration j gives:
#   - Primal feasibility with observed coal_exch^j (from proposed_coals)
#   - Strong duality with known lam^j, ct^j
#   - Complementary slackness

function solve_coal_inverse_trajectory(
        traj::ADMMTrajectory, fr::CoalForwardResult, agent::Int;
        objective_var::Symbol, objective_t::Int = 1, sense::Symbol = :min,
        n_iters_use::Int = 10, known_battery::Bool = false, tol::Float64 = 1e-4,
        cs_tol::Float64 = Inf)

    h = traj.h
    eta_ch = fr.eta_ch
    eta_dis = fr.eta_dis
    pb = fr.pb
    ps = fr.ps
    Q_true = fr.Q_b[agent]
    P_true = fr.P_b[agent]
    SoC0_true = fr.SoC0[agent]
    c = 0.5  # ADMM penalty parameter

    n_iters_use = min(n_iters_use, traj.n_iters)
    if n_iters_use >= traj.n_iters
        iter_inds = collect(1:traj.n_iters)
    elseif n_iters_use == 1
        iter_inds = [traj.n_iters]
    else
        iter_inds = round.(Int, range(1, traj.n_iters, length=n_iters_use))
    end
    J = length(iter_inds)

    m = Model(HiGHS.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "presolve", "on")

    # Shared private parameters
    @variable(m, net_load[1:h])
    if known_battery
        Q_b = Q_true
        P_b = P_true
        @variable(m, SoC_0 == SoC0_true)
    else
        @variable(m, 0 <= Q_b <= Q_MAX)
        @variable(m, 0 <= P_b <= P_MAX)
        @variable(m, 0 <= SoC_0)
    end

    # Per-iteration primal variables
    g_buy  = Vector{Vector{VariableRef}}(undef, J)
    g_sell = Vector{Vector{VariableRef}}(undef, J)
    pos_d  = Vector{Vector{VariableRef}}(undef, J)
    neg_d  = Vector{Vector{VariableRef}}(undef, J)
    charge = Vector{Vector{VariableRef}}(undef, J)
    for j in 1:J
        g_buy[j]  = @variable(m, [t=1:h], lower_bound=0, base_name="gb_$(j)_t")
        g_sell[j] = @variable(m, [t=1:h], lower_bound=0, base_name="gs_$(j)_t")
        pos_d[j]  = @variable(m, [t=1:h], lower_bound=0, base_name="pd_$(j)_t")
        neg_d[j]  = @variable(m, [t=1:h], upper_bound=0, base_name="nd_$(j)_t")
        charge[j] = @variable(m, [t=1:h], lower_bound=0, base_name="ch_$(j)_t")
    end

    for j in 1:J
        iter = iter_inds[j]
        ce_j = traj.proposed_coals_hist[iter][:,agent]
        z_j = traj.z_hist[iter][:,agent]  # observed grid trade at this iteration

        # Primal feasibility: net_load + g_sell + pos_d + neg_d + ce_j = g_buy
        # Observed: g_buy - g_sell = z_j (grid trade observed per iteration)
        for t in 1:h
            @constraint(m, g_buy[j][t] - g_sell[j][t] == z_j[t])
            @constraint(m, net_load[t] + g_sell[j][t] + pos_d[j][t] + neg_d[j][t] + ce_j[t] == g_buy[j][t])
        end

        # Battery dynamics
        @constraint(m, charge[j][1] == SoC_0)
        for t in 2:h
            @constraint(m, charge[j][t] == charge[j][t-1] + eta_ch * pos_d[j][t-1] + (1/eta_dis) * neg_d[j][t-1])
        end

        # Capacity
        for t in 1:h
            @constraint(m, pos_d[j][t] <= P_b)
            @constraint(m, -neg_d[j][t] <= P_b)
            @constraint(m, charge[j][t] <= Q_b)
        end
        @constraint(m, pos_d[j][h] + neg_d[j][h] + charge[j][h] >= 0)
    end

    @constraint(m, nl_lo[t=1:h], net_load[t] >= -NL_MAX)
    @constraint(m, nl_hi[t=1:h], net_load[t] <= NL_MAX)
    if !known_battery
        @constraint(m, soc_cap, SoC_0 <= Q_b)
    end

    # --- Model C: primal feasibility + per-iteration strong duality ---
    # Each ADMM iteration j solves a QP. When coal_exch is observed (= ce_j), the
    # quadratic term becomes a known constant K_j, reducing the QP to an LP.
    # LP strong duality gives a linear equation in net_load, Q_b, P_b, SoC_0.
    # Different iterations give different equations (different mu_j, sigma_j, etc.)
    # that tighten the feasible set.
    sd_tol = max(0.5, 1e-3 * abs(fr.primal_obj))  # wider for multi-iteration

    for j in 1:J
        iter = iter_inds[j]
        ce_j = traj.proposed_coals_hist[iter][:,agent]
        z_j = traj.z_hist[iter][:,agent]
        lam_j = traj.lambdas_hist[iter][:,agent]
        ct_j = iter > 1 ? traj.new_coals_hist[iter-1][:,agent] : zeros(h)
        K_j = dot(lam_j, ce_j) + (c/2) * sum((ce_j .- ct_j).^2)

        mu_j = traj.mu_hist[iter][:,agent]
        sigma_j = traj.sigma_hist[iter][:,agent]
        alpha_j = traj.alpha_hist[iter][:,agent]
        beta_j = traj.beta_hist[iter][:,agent]
        lambda_init_j = traj.lambda_init_hist[iter, agent]

        # Strong duality: primal = dual
        # primal_j = sum(pb*gb_j - ps*gs_j) + K_j
        # dual_j   = -sum(net_load*mu_j) + SoC_0*lambda_init_j + Q_b*sum(sigma_j) + P_b*sum(alpha_j) + P_b*sum(beta_j)
        primal_j = sum(pb[t]*g_buy[j][t] - ps[t]*g_sell[j][t] for t in 1:h) + K_j
        dual_j = -sum(net_load[t]*mu_j[t] for t in 1:h) + SoC_0*lambda_init_j + Q_b*sum(sigma_j) + P_b*sum(alpha_j) + P_b*sum(beta_j)

        @constraint(m, primal_j - dual_j <= sd_tol)
        @constraint(m, dual_j - primal_j <= sd_tol)

        # Complementary slackness: |dual| > tol => constraint active.
        # Use tolerance bands instead of exact equality — SCS duals at 1e-4
        # tolerance give ~0.3 absolute error in the primal, so exact CS
        # constraints are too tight across multiple iterations.
        # NOTE: CS is disabled by default (cs_tol = Inf) because SCS dual
        # noise makes it infeasible across multiple iterations. The tightening
        # from SD + multiple z_j observations is the main signal.
        for t in 1:h
            if abs(sigma_j[t]) > tol && cs_tol < Inf
                @constraint(m, charge[j][t] <= Q_b)
                @constraint(m, charge[j][t] >= Q_b - cs_tol)
            end
            if abs(alpha_j[t]) > tol && cs_tol < Inf
                @constraint(m, pos_d[j][t] <= P_b)
                @constraint(m, pos_d[j][t] >= P_b - cs_tol)
            end
            if abs(beta_j[t]) > tol && cs_tol < Inf
                @constraint(m, -neg_d[j][t] <= P_b)
                @constraint(m, -neg_d[j][t] >= P_b - cs_tol)
            end
        end
    end

    # Objective
    if objective_var == :net_load
        obj = net_load[objective_t]
    elseif objective_var == :Q_b
        known_battery && return Q_true, "FIXED"
        obj = Q_b
    elseif objective_var == :P_b
        known_battery && return P_true, "FIXED"
        obj = P_b
    elseif objective_var == :SoC_0
        known_battery && return SoC0_true, "FIXED"
        obj = SoC_0
    else
        error("Unknown objective_var: $objective_var")
    end

    sense == :min ? @objective(m, Min, obj) : @objective(m, Max, obj)
    optimize!(m)

    status = string(termination_status(m))
    val = (status == "OPTIMAL" || primal_status(m) == MOI.FEASIBLE_POINT) ? value(obj) : NaN
    return val, status
end

# ===========================================================================
# Helper: compute all bounds for one agent
# ===========================================================================

function compute_coal_bounds(fr::CoalForwardResult, traj::ADMMTrajectory, agent::Int;
        known_battery::Bool = false)
    h = fr.h
    results = DataFrame(
        param = String[], t = Int[],
        min_A = Float64[], max_A = Float64[],
        min_B = Float64[], max_B = Float64[],
        min_C5 = Float64[], max_C5 = Float64[],
        min_C10 = Float64[], max_C10 = Float64[],
        min_C20 = Float64[], max_C20 = Float64[],
        theta_true = Float64[], ratio_A = Float64[], ratio_B = Float64[],
        ratio_C5 = Float64[], ratio_C10 = Float64[], ratio_C20 = Float64[],
    )

    for t in 1:h
        a_min, _ = solve_coal_inverse(fr, agent; objective_var=:net_load, objective_t=t, sense=:min, model=:A, known_battery=known_battery)
        a_max, _ = solve_coal_inverse(fr, agent; objective_var=:net_load, objective_t=t, sense=:max, model=:A, known_battery=known_battery)
        b_min, _ = solve_coal_inverse(fr, agent; objective_var=:net_load, objective_t=t, sense=:min, model=:B, known_battery=known_battery)
        b_max, _ = solve_coal_inverse(fr, agent; objective_var=:net_load, objective_t=t, sense=:max, model=:B, known_battery=known_battery)
        c5_min, _ = solve_coal_inverse_trajectory(traj, fr, agent; objective_var=:net_load, objective_t=t, sense=:min, n_iters_use=5, known_battery=known_battery)
        c5_max, _ = solve_coal_inverse_trajectory(traj, fr, agent; objective_var=:net_load, objective_t=t, sense=:max, n_iters_use=5, known_battery=known_battery)
        c10_min, _ = solve_coal_inverse_trajectory(traj, fr, agent; objective_var=:net_load, objective_t=t, sense=:min, n_iters_use=10, known_battery=known_battery)
        c10_max, _ = solve_coal_inverse_trajectory(traj, fr, agent; objective_var=:net_load, objective_t=t, sense=:max, n_iters_use=10, known_battery=known_battery)
        c20_min, _ = solve_coal_inverse_trajectory(traj, fr, agent; objective_var=:net_load, objective_t=t, sense=:min, n_iters_use=20, known_battery=known_battery)
        c20_max, _ = solve_coal_inverse_trajectory(traj, fr, agent; objective_var=:net_load, objective_t=t, sense=:max, n_iters_use=20, known_battery=known_battery)

        theta = fr.net_load_true[t,agent]
        denom = max(abs(theta), 1.0)
        push!(results, ("net_load", t, a_min, a_max, b_min, b_max,
                        c5_min, c5_max, c10_min, c10_max, c20_min, c20_max,
                        theta,
                        (a_max-a_min)/denom, (b_max-b_min)/denom,
                        (c5_max-c5_min)/denom, (c10_max-c10_min)/denom, (c20_max-c20_min)/denom))
    end

    return results
end

# ===========================================================================
# Main analysis
# ===========================================================================

function main()
    num_builds = 5
    num_steps = 96
    all_buildings, energy_cost, energy_sale = MPC_load_from_CSV(num_builds, num_steps)
    global opt = MPC_optimiser(energy_cost', energy_sale')

    k = 20
    h = 16

    # 2-agent coalition: buildings 1 (large batt) and 3 (small batt)
    bs = [all_buildings[1], all_buildings[3]]
    println("Coalition: buildings 1 and 3")
    println("  Bldg 1: Q_b=$(bs[1].max_storage), P_b=$(bs[1].storage_max_flow)")
    println("  Bldg 3: Q_b=$(bs[2].max_storage), P_b=$(bs[2].storage_max_flow)")

    # Forward solve: centralized LP (gives clean duals + a valid optimal solution)
    fr = coal_forward_solve(bs, k, h, opt)
    println("\nCentralized LP obj = $(round(fr.primal_obj, digits=4))")
    for i in 1:fr.n_agents
        println("  Agent $i: z* = $(round.(fr.z_star[:,i], digits=3))")
        println("           ce = $(round.(fr.coal_exch[:,i], digits=3))")
    end

    # ADMM with trajectory (for Model C — uses ADMM's own mu_j computation,
    # does NOT require centralized duals)
    println("\nRunning ADMM with trajectory recording...")
    traj = run_admm_with_trajectory(bs, k, h, opt; abs_tol=1e-8)
    println("ADMM converged in $(traj.n_iters) iterations")
    println("  ADMM z1 = $(round.(traj.z_star[:,1], digits=3))")
    println("  ADMM ce1 = $(round.(traj.coal_exch[:,1], digits=3))")
    println("  Central z1 = $(round.(fr.z_star[:,1], digits=3))")
    println("  Central ce1 = $(round.(fr.coal_exch[:,1], digits=3))")
    println("  (Note: ADMM and centralized LP may give different optimal vertices)")
    println("  (Models A/B use centralized duals; Model C uses per-iteration ADMM duals)")

    # Compute bounds for each agent
    for agent in 1:fr.n_agents
        println("\nComputing bounds for agent $agent (building $(bs[agent].id))...")
        res = compute_coal_bounds(fr, traj, agent; known_battery=true)

        # Print table
        println("\n=== Agent $agent: net_load Observability (known battery) ===")
        println("=" ^ 140)
        @printf("%-5s %4s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
            "Param", "t", "min_A", "max_A", "min_B", "max_B",
            "min_C5", "max_C5", "min_C10", "max_C10", "min_C20", "max_C20", "true")
        println("-" ^ 140)
        for row in eachrow(res)
            @printf("%-5s %4d %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n",
                row.param, row.t,
                row.min_A, row.max_A, row.min_B, row.max_B,
                row.min_C5, row.max_C5, row.min_C10, row.max_C10,
                row.min_C20, row.max_C20, row.theta_true)
        end
        println("=" ^ 140)
        @printf("Mean ratios:  A=%.2f  B=%.2f  C5=%.2f  C10=%.2f  C20=%.2f\n",
            mean(res.ratio_A), mean(res.ratio_B), mean(res.ratio_C5),
            mean(res.ratio_C10), mean(res.ratio_C20))

        CSV.write("results/observability_coal_agent$(agent)_unknown.csv", res)
    end

    # Plot: per-timestep bounds for agent 1
    for agent in 1:fr.n_agents
        res_df = CSV.read("results/observability_coal_agent$(agent)_unknown.csv", DataFrame)
        p = plot(1:h, res_df.theta_true, label="true", lw=2, color=:black)
        plot!(1:h, res_df.min_A, fillrange=res_df.max_A, alpha=0.15, color=:blue, label="Model A (z only)")
        plot!(1:h, res_df.min_B, fillrange=res_df.max_B, alpha=0.15, color=:green, label="Model B (z+ce)")
        plot!(1:h, res_df.min_C10, fillrange=res_df.max_C10, alpha=0.3, color=:red, label="Model C (10 iters)")
        xlabel!("timestep t"); ylabel!("net_load (kW)")
        title!("Agent $agent: net_load observability in coalition")
        savefig(p, "results/observability_coal_agent$(agent).pdf")
        println("Saved: results/observability_coal_agent$(agent).pdf")
    end

    # Summary plot: mean ratio by model and agent
    summary = DataFrame(
        agent = Int[], model = String[], mean_ratio = Float64[]
    )
    for agent in 1:fr.n_agents
        res = CSV.read("results/observability_coal_agent$(agent)_unknown.csv", DataFrame)
        push!(summary, (agent, "A (z only)", mean(res.ratio_A)))
        push!(summary, (agent, "B (z+ce)", mean(res.ratio_B)))
        push!(summary, (agent, "C (5 iters)", mean(res.ratio_C5)))
        push!(summary, (agent, "C (10 iters)", mean(res.ratio_C10)))
        push!(summary, (agent, "C (20 iters)", mean(res.ratio_C20)))
    end
    println("\n=== Summary: Mean Observability Ratios ===")
    println(summary)

    p_sum = plot()
    for agent in 1:fr.n_agents
        sub = summary[summary.agent .== agent, :]
        plot!(p_sum, 1:5, sub.mean_ratio, marker=:o, label="Agent $agent", lw=2)
    end
    xticks!(p_sum, (1:5, ["A", "B", "C5", "C10", "C20"]))
    xlabel!(p_sum, "Observation model"); ylabel!(p_sum, "Mean observability ratio")
    title!(p_sum, "Coalition observability comparison")
    savefig(p_sum, "results/observability_coal_summary.pdf")
    println("Saved: results/observability_coal_summary.pdf")

    CSV.write("results/observability_coal_summary.csv", summary)

    return fr, traj
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
