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
# KKT-MILP: banded complementary slackness block builder
# ===========================================================================
# One block = one KKT system for one observation (z_obs, ce_obs).
# Shared theta (net_load, SoC_0) referenced across blocks. Battery fixed (known).
# Banded CS: exact equalities relaxed to tolerance bands (cs_tol) to accommodate
# SCS solver noise on the ADMM trajectory data.

function add_kkt_block!(m, net_load, SoC_0, z_obs, ce_obs, h, eta_ch, eta_dis,
                        pb, ps, j::Int, Q_b_val, P_b_val, cs_tol::Float64)
    z_tol = 1e-3
    pb_max = maximum(pb)
    # λ bounds: marginal value of stored energy. Wider than singleton to
    # accommodate ADMM-perturbed stationarity (mu-al)/eta can reach -pb_max/eta.
    lam_lo = -pb_max / eta_ch - 0.05
    lam_hi =  pb_max / eta_ch + 0.05
    dual_hi = pb_max + 0.05
    sfx = "b$j"

    # --- Primal variables ---
    gb  = @variable(m, [1:h], lower_bound=0, upper_bound=NL_MAX, base_name="gb_$sfx")
    gs  = @variable(m, [1:h], lower_bound=0, upper_bound=NL_MAX, base_name="gs_$sfx")
    pos = @variable(m, [1:h], lower_bound=0, upper_bound=P_b_val, base_name="pos_$sfx")
    neg = @variable(m, [1:h], lower_bound=-P_b_val, upper_bound=0, base_name="neg_$sfx")
    ch  = @variable(m, [1:h], lower_bound=0, upper_bound=Q_b_val, base_name="ch_$sfx")

    # --- Dual variables ---
    mu  = @variable(m, [1:h], base_name="mu_$sfx")
    for t in 1:h
        set_lower_bound(mu[t], -pb[t]); set_upper_bound(mu[t], -ps[t])
    end
    lam_init = @variable(m, lower_bound=lam_lo, upper_bound=lam_hi, base_name="li_$sfx")
    lam_dyn  = @variable(m, [2:h], lower_bound=lam_lo, upper_bound=lam_hi, base_name="ld_$sfx")
    sig = @variable(m, [1:h], lower_bound=0, upper_bound=dual_hi, base_name="sg_$sfx")
    alp = @variable(m, [1:h], lower_bound=0, upper_bound=dual_hi, base_name="al_$sfx")
    bet = @variable(m, [1:h], lower_bound=0, upper_bound=dual_hi, base_name="be_$sfx")
    php = @variable(m, lower_bound=0, upper_bound=dual_hi, base_name="ph_$sfx")

    # --- z* fixing: pin g_buy/g_sell/mu at trading timesteps ---
    z_eff = [abs(z_obs[t]) <= z_tol ? 0.0 : z_obs[t] for t in 1:h]
    for t in 1:h
        if z_eff[t] > z_tol
            fix(gs[t], 0.0; force=true)
            @constraint(m, mu[t] == -pb[t])
        elseif z_eff[t] < -z_tol
            fix(gb[t], 0.0; force=true)
            @constraint(m, mu[t] == -ps[t])
        else
            fix(gb[t], 0.0; force=true)
            fix(gs[t], 0.0; force=true)
        end
    end

    # --- Primal feasibility (power balance includes + ce_obs) ---
    for t in 1:h
        @constraint(m, gb[t] - gs[t] == z_eff[t])
        @constraint(m, net_load[t] + gs[t] + pos[t] + neg[t] + ce_obs[t] == gb[t])
    end
    @constraint(m, ch[1] == SoC_0)
    for t in 2:h
        @constraint(m, ch[t] == ch[t-1] + eta_ch*pos[t-1] + (1/eta_dis)*neg[t-1])
    end
    for t in 1:h
        @constraint(m, pos[t] <= P_b_val)
        @constraint(m, -neg[t] <= P_b_val)
        @constraint(m, ch[t] <= Q_b_val)
    end
    @constraint(m, pos[h] + neg[h] + ch[h] >= 0)

    # --- Banded complementary slackness (indicator constraints) ---
    # Non-neg CS: y=1 => primal <= cs_tol (at bound within tol); y=0 => tau == 0
    # Capacity CS: y=1 => dual == 0; y=0 => primal within cs_tol of cap
    y_pos = @variable(m, [1:h], binary=true, base_name="yp_$sfx")
    y_neg = @variable(m, [1:h], binary=true, base_name="yn_$sfx")
    y_ch  = @variable(m, [1:h], binary=true, base_name="yc_$sfx")
    y_sg  = @variable(m, [1:h], binary=true, base_name="ys_$sfx")
    y_al  = @variable(m, [1:h], binary=true, base_name="ya_$sfx")
    y_be  = @variable(m, [1:h], binary=true, base_name="yb_$sfx")
    y_ph  = @variable(m, binary=true, base_name="yf_$sfx")

    for t in 1:h
        # pos_δ non-neg CS
        @constraint(m, y_pos[t] => {pos[t] <= cs_tol})
        if t < h
            @constraint(m, !y_pos[t] => {-mu[t] + eta_ch*lam_dyn[t+1] + alp[t] == 0})
            @constraint(m, -mu[t] + eta_ch*lam_dyn[t+1] + alp[t] >= 0)
        else
            @constraint(m, !y_pos[t] => {-mu[t] - php + alp[t] == 0})
            @constraint(m, -mu[t] - php + alp[t] >= 0)
        end

        # neg_δ non-neg CS (neg <= 0, upper bound at 0)
        @constraint(m, y_neg[t] => {-neg[t] <= cs_tol})
        if t < h
            @constraint(m, !y_neg[t] => {mu[t] - lam_dyn[t+1]/eta_dis + bet[t] == 0})
            @constraint(m, mu[t] - lam_dyn[t+1]/eta_dis + bet[t] >= 0)
        else
            @constraint(m, !y_neg[t] => {mu[t] + php + bet[t] == 0})
            @constraint(m, mu[t] + php + bet[t] >= 0)
        end

        # charge non-neg CS
        @constraint(m, y_ch[t] => {ch[t] <= cs_tol})
        if t == 1
            @constraint(m, !y_ch[t] => {-lam_init + lam_dyn[2] + sig[1] == 0})
            @constraint(m, -lam_init + lam_dyn[2] + sig[1] >= 0)
        elseif t < h
            @constraint(m, !y_ch[t] => {-lam_dyn[t] + lam_dyn[t+1] + sig[t] == 0})
            @constraint(m, -lam_dyn[t] + lam_dyn[t+1] + sig[t] >= 0)
        else
            @constraint(m, !y_ch[t] => {-lam_dyn[t] - php + sig[t] == 0})
            @constraint(m, -lam_dyn[t] - php + sig[t] >= 0)
        end

        # Capacity CS (banded)
        @constraint(m, y_sg[t] => {sig[t] == 0})
        @constraint(m, !y_sg[t] => {ch[t] >= Q_b_val - cs_tol})
        @constraint(m, y_al[t] => {alp[t] == 0})
        @constraint(m, !y_al[t] => {pos[t] >= P_b_val - cs_tol})
        @constraint(m, y_be[t] => {bet[t] == 0})
        @constraint(m, !y_be[t] => {-neg[t] >= P_b_val - cs_tol})
    end
    @constraint(m, y_ph => {php == 0})
    @constraint(m, !y_ph => {pos[h] + neg[h] + ch[h] <= cs_tol})
end

# ===========================================================================
# KKT-MILP solvers: Models B (converged) and C (trajectory)
# ===========================================================================
# Model B: single observation (z_star, coal_exch from centralized LP)
# Model C: multiple observations (z_j, ce_j from ADMM trajectory)
# Both use banded CS with cs_tol. Model B uses tight band (centralized, 1e-6 data);
# Model C uses wider band (ADMM, 1e-4 data).

function solve_coal_milp(observations::Vector{Tuple{Vector{Float64}, Vector{Float64}}},
                         fr::CoalForwardResult, agent::Int;
                         objective_t::Int, sense::Symbol,
                         cs_tol::Float64 = 0.5, time_limit::Float64 = 120.0)
    h = fr.h; eta_ch = fr.eta_ch; eta_dis = fr.eta_dis
    pb = fr.pb; ps = fr.ps
    Q_b_val = fr.Q_b[agent]; P_b_val = fr.P_b[agent]; SoC0_val = fr.SoC0[agent]

    m = Model(HiGHS.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "presolve", "on")
    set_optimizer_attribute(m, "time_limit", time_limit)

    @variable(m, -NL_MAX <= net_load[1:h] <= NL_MAX)
    @variable(m, SoC_0 == SoC0_val)

    for (j, (zk, cek)) in enumerate(observations)
        add_kkt_block!(m, net_load, SoC_0, zk, cek, h, eta_ch, eta_dis,
                       pb, ps, j, Q_b_val, P_b_val, cs_tol)
    end

    obj = net_load[objective_t]
    sense == :min ? @objective(m, Min, obj) : @objective(m, Max, obj)
    optimize!(m)

    status = string(termination_status(m))
    val = (status == "OPTIMAL" || primal_status(m) == MOI.FEASIBLE_POINT) ? value(obj) : NaN
    return val, status
end

# ===========================================================================
# Helper: compute coalition bounds for one agent (KKT-MILP, all models)
# ===========================================================================

function compute_coal_bounds(fr::CoalForwardResult, traj::ADMMTrajectory, agent::Int)
    h = fr.h
    results = DataFrame(
        param = String[], t = Int[],
        min_B = Float64[], max_B = Float64[],
        min_C5 = Float64[], max_C5 = Float64[],
        min_C10 = Float64[], max_C10 = Float64[],
        theta_true = Float64[],
        width_B = Float64[], width_C5 = Float64[], width_C10 = Float64[],
        ratio_B = Float64[], ratio_C5 = Float64[], ratio_C10 = Float64[],
    )

    # Model B observations: converged (z*, ce*) from centralized LP
    obs_B = [(fr.z_star[:, agent], fr.coal_exch[:, agent])]
    cs_B = 0.05  # tight band: centralized LP at 1e-6 tolerance

    # Model C observations: ADMM trajectory
    n = traj.n_iters
    iter_inds_5  = n >= 5  ? round.(Int, range(1, n, length=min(5, n)))  : collect(1:n)
    iter_inds_10 = n >= 10 ? round.(Int, range(1, n, length=min(10, n))) : collect(1:n)
    obs_C5  = [(traj.z_hist[it][:, agent], traj.proposed_coals_hist[it][:, agent]) for it in iter_inds_5]
    obs_C10 = [(traj.z_hist[it][:, agent], traj.proposed_coals_hist[it][:, agent]) for it in iter_inds_10]
    cs_C = 0.5  # wider band: ADMM at 1e-4 tolerance

    for t in 1:h
        b_min, _  = solve_coal_milp(obs_B,  fr, agent; objective_t=t, sense=:min, cs_tol=cs_B, time_limit=60.0)
        b_max, _  = solve_coal_milp(obs_B,  fr, agent; objective_t=t, sense=:max, cs_tol=cs_B, time_limit=60.0)
        c5_min, _ = solve_coal_milp(obs_C5,  fr, agent; objective_t=t, sense=:min, cs_tol=cs_C, time_limit=60.0)
        c5_max, _ = solve_coal_milp(obs_C5,  fr, agent; objective_t=t, sense=:max, cs_tol=cs_C, time_limit=60.0)
        c10_min, _ = solve_coal_milp(obs_C10, fr, agent; objective_t=t, sense=:min, cs_tol=cs_C, time_limit=60.0)
        c10_max, _ = solve_coal_milp(obs_C10, fr, agent; objective_t=t, sense=:max, cs_tol=cs_C, time_limit=60.0)

        theta = fr.net_load_true[t, agent]
        denom = max(abs(theta), 1.0)
        push!(results, ("net_load", t, b_min, b_max, c5_min, c5_max, c10_min, c10_max, theta,
            (b_max - b_min), (c5_max - c5_min), (c10_max - c10_min),
            (b_max - b_min) / denom, (c5_max - c5_min) / denom, (c10_max - c10_min) / denom))
    end

    return results
end

# ===========================================================================
# Main analysis (KKT-MILP, all models)
# ===========================================================================

function main()
    num_builds = 5
    num_steps = 96
    all_buildings, energy_cost, energy_sale = MPC_load_from_CSV(num_builds, num_steps)
    global opt = MPC_optimiser(energy_cost', energy_sale')

    k = 20
    h = 16

    bs = [all_buildings[1], all_buildings[3]]
    println("Coalition: buildings 1 and 3")
    println("  Bldg 1: Q_b=$(bs[1].max_storage), P_b=$(bs[1].storage_max_flow)")
    println("  Bldg 3: Q_b=$(bs[2].max_storage), P_b=$(bs[2].storage_max_flow)")

    fr = coal_forward_solve(bs, k, h, opt)
    println("\nCentralized LP obj = $(round(fr.primal_obj, digits=4))")
    for i in 1:fr.n_agents
        println("  Agent $i: z* = $(round.(fr.z_star[:,i], digits=3))")
        println("           ce = $(round.(fr.coal_exch[:,i], digits=3))")
    end

    println("\nRunning ADMM with trajectory recording...")
    traj = run_admm_with_trajectory(bs, k, h, opt; abs_tol=1e-8)
    println("ADMM converged in $(traj.n_iters) iterations")

    # Compute KKT-MILP bounds for each agent
    for agent in 1:fr.n_agents
        println("\nComputing KKT-MILP bounds for agent $agent (building $(bs[agent].id))...")
        res = compute_coal_bounds(fr, traj, agent)

        println("\n=== Agent $agent: net_load Observability (KKT-MILP, known battery) ===")
        println("=" ^ 130)
        @printf("%-5s %4s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
            "Param", "t", "min_B", "max_B", "min_C5", "max_C5",
            "min_C10", "max_C10", "true", "w_B", "w_C5", "w_C10")
        println("-" ^ 130)
        for row in eachrow(res)
            @printf("%-5s %4d %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.1f %10.1f %10.1f\n",
                row.param, row.t,
                row.min_B, row.max_B, row.min_C5, row.max_C5,
                row.min_C10, row.max_C10, row.theta_true,
                row.width_B, row.width_C5, row.width_C10)
        end
        println("=" ^ 130)
        @printf("Mean ratios:  B=%.2f  C5=%.2f  C10=%.2f\n",
            mean(res.ratio_B), mean(res.ratio_C5), mean(res.ratio_C10))

        CSV.write("results/observability_coal_agent$(agent)_milp.csv", res)
    end

    # Plot: per-timestep bounds
    for agent in 1:fr.n_agents
        res_df = CSV.read("results/observability_coal_agent$(agent)_milp.csv", DataFrame)
        p = plot(1:h, res_df.theta_true, label="true", lw=2, color=:black)
        plot!(1:h, res_df.min_B, fillrange=res_df.max_B, alpha=0.2, color=:green, label="Model B (converged)")
        plot!(1:h, res_df.min_C10, fillrange=res_df.max_C10, alpha=0.3, color=:red, label="Model C (10 iters)")
        xlabel!("timestep t"); ylabel!("net_load (kW)")
        title!("Agent $agent: net_load observability (KKT-MILP)")
        savefig(p, "results/observability_coal_agent$(agent).pdf")
        println("Saved: results/observability_coal_agent$(agent).pdf")
    end

    # Summary
    summary = DataFrame(
        agent = Int[], model = String[], mean_ratio = Float64[], mean_width = Float64[]
    )
    for agent in 1:fr.n_agents
        res = CSV.read("results/observability_coal_agent$(agent)_milp.csv", DataFrame)
        push!(summary, (agent, "B (converged)", mean(res.ratio_B), mean(res.width_B)))
        push!(summary, (agent, "C (5 iters)", mean(res.ratio_C5), mean(res.width_C5)))
        push!(summary, (agent, "C (10 iters)", mean(res.ratio_C10), mean(res.width_C10)))
    end
    println("\n=== Summary: Mean Observability Ratios (KKT-MILP) ===")
    println(summary)

    p_sum = plot()
    for agent in 1:fr.n_agents
        sub = summary[summary.agent .== agent, :]
        plot!(p_sum, 1:3, sub.mean_ratio, marker=:o, label="Agent $agent", lw=2)
    end
    xticks!(p_sum, (1:3, ["B", "C5", "C10"]))
    xlabel!(p_sum, "Observation model"); ylabel!(p_sum, "Mean observability ratio")
    title!(p_sum, "Coalition observability (KKT-MILP)")
    savefig(p_sum, "results/observability_coal_summary.pdf")
    println("Saved: results/observability_coal_summary.pdf")

    CSV.write("results/observability_coal_summary.csv", summary)

    return fr, traj
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
