# test_kkt.jl — Test the KKT-based MILP (Variant C) against Variants A and B.
# Synthetic problem: no data files needed.
#
# Run: julia test_kkt.jl

include("observability.jl")
using JuMP, HiGHS, LinearAlgebra, Printf, Test

# ===========================================================================
# 1. Synthetic forward problem (h=4, price change at t=3)
# ===========================================================================

h = 4
pb = [0.15, 0.15, 0.17, 0.17]   # buy prices
ps = [0.10, 0.10, 0.12, 0.12]   # sell prices
η_ch = 0.95
η_dis = 0.95
Q_b_true = 100.0
P_b_true = 25.0
SoC0_true = 20.0
nl_true = [10.0, 10.0, 10.0, 10.0]

# Forward solve: direct JuMP model (no MPC_Building dependency)
function synthetic_forward(nl, Q_b, P_b, SoC0, pb, ps, η_ch, η_dis, h)
    model = Model(SCS.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "eps_abs", 1e-6)
    set_optimizer_attribute(model, "eps_rel", 1e-6)

    @variable(model, g_buy[1:h] >= 0)
    @variable(model, g_sell[1:h] >= 0)
    @variable(model, pos_δ[1:h] >= 0)
    @variable(model, neg_δ[1:h] <= 0)
    @variable(model, charge[1:h] >= 0)

    @constraint(model, power_c[t=1:h],
        nl[t] + g_sell[t] + pos_δ[t] + neg_δ[t] == g_buy[t])
    @constraint(model, init_c, charge[1] == SoC0)
    @constraint(model, dyn_c[t=2:h],
        charge[t] == charge[t-1] + η_ch * pos_δ[t-1] + (1/η_dis) * neg_δ[t-1])
    @constraint(model, pos_cap[t=1:h], pos_δ[t] <= P_b)
    @constraint(model, neg_cap[t=1:h], -neg_δ[t] <= P_b)
    @constraint(model, charge_cap[t=1:h], charge[t] <= Q_b)
    @constraint(model, final_c, pos_δ[h] + neg_δ[h] + charge[h] >= 0)

    @objective(model, Min, sum(pb[t] * g_buy[t] - ps[t] * g_sell[t] for t in 1:h))
    optimize!(model)

    @assert termination_status(model) == MOI.OPTIMAL "Forward solve failed"

    z_star = value.(g_buy) .- value.(g_sell)
    μ = dual.(power_c)
    λ_init = dual(init_c)
    λ_dyn = h > 1 ? [dual(dyn_c[t]) for t in 2:h] : Float64[]
    σ_hi = dual.(charge_cap)
    α_prime = dual.(pos_cap)
    β_prime = dual.(neg_cap)
    φ = dual(final_c)
    pobj = objective_value(model)
    dobj = dual_objective_value(model)

    return ForwardResult(z_star, value.(g_buy), value.(g_sell),
        value.(pos_δ), value.(neg_δ), value.(charge),
        μ, λ_init, λ_dyn, σ_hi, α_prime, β_prime, φ,
        nl, Q_b, P_b, SoC0, pb, ps, η_ch, η_dis, pobj, dobj, h)
end

fr = synthetic_forward(nl_true, Q_b_true, P_b_true, SoC0_true,
    pb, ps, η_ch, η_dis, h)

println("Forward solve complete.")
println("  z* = $(round.(fr.z_star, digits=3))")
println("  net_load_true = $(nl_true)")
println("  Q_b=$Q_b_true, P_b=$P_b_true, SoC_0=$SoC0_true")
println("  primal_obj = $(round(fr.primal_obj, digits=4))")

# ===========================================================================
# 2. Solve all three variants and compare
# ===========================================================================

TOL = 1e-3

println("\n" * "=" ^ 100)
println("Variant comparison (h=$h)")
println("=" ^ 100)
@printf("%-12s %4s %12s %12s %12s %12s %12s %12s\n",
    "Param", "t", "min_A", "max_A", "min_B", "max_B", "min_C", "max_C")
println("-" ^ 100)

all_pass = true

# --- net_load[t] ---
for t in 1:h
    a_min, _ = solve_inverse_feasibility(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=:net_load, objective_t=t, sense=:min, known_battery=false)
    a_max, _ = solve_inverse_feasibility(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=:net_load, objective_t=t, sense=:max, known_battery=false)
    b_min, _ = solve_inverse_optimality(fr;
        objective_var=:net_load, objective_t=t, sense=:min, known_battery=false)
    b_max, _ = solve_inverse_optimality(fr;
        objective_var=:net_load, objective_t=t, sense=:max, known_battery=false)
    c_min, _ = solve_inverse_optimality_kkt(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=:net_load, objective_t=t, sense=:min, known_battery=false)
    c_max, _ = solve_inverse_optimality_kkt(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=:net_load, objective_t=t, sense=:max, known_battery=false)

    θ_true = nl_true[t]
    @printf("%-12s %4d %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n",
        "net_load", t, a_min, a_max, b_min, b_max, c_min, c_max)

    # Truth containment (C)
    if !(c_min - TOL <= θ_true <= c_max + TOL)
        println("  FAIL: truth containment C: $θ_true not in [$c_min, $c_max]")
        global all_pass = false
    end
    # Nesting A ⊇ C (guaranteed: KKT implies primal feasibility)
    if !(a_min - TOL <= c_min + TOL && c_max - TOL <= a_max + TOL)
        println("  FAIL: nesting A ⊇ C: A=[$a_min,$a_max] C=[$c_min,$c_max]")
        global all_pass = false
    end
    # B vs C: not guaranteed to nest (B relaxes SD, C fixes it exactly).
    # Report if B is wider than C (expected: B's SD relaxation).
    if b_min < c_min - TOL || b_max > c_max + TOL
        println("  NOTE: B wider than C (expected, SD relaxation): B=[$b_min,$b_max] C=[$c_min,$c_max]")
    end
end

# --- Q_b, P_b, SoC_0 ---
for param in (:Q_b, :P_b, :SoC_0)
    a_min, _ = solve_inverse_feasibility(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=param, sense=:min, known_battery=false)
    a_max, _ = solve_inverse_feasibility(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=param, sense=:max, known_battery=false)
    b_min, _ = solve_inverse_optimality(fr;
        objective_var=param, sense=:min, known_battery=false)
    b_max, _ = solve_inverse_optimality(fr;
        objective_var=param, sense=:max, known_battery=false)
    c_min, _ = solve_inverse_optimality_kkt(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=param, sense=:min, known_battery=false)
    c_max, _ = solve_inverse_optimality_kkt(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=param, sense=:max, known_battery=false)

    θ_true = param == :Q_b ? Q_b_true : param == :P_b ? P_b_true : SoC0_true
    @printf("%-12s %4d %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n",
        string(param), 0, a_min, a_max, b_min, b_max, c_min, c_max)

    if !(c_min - TOL <= θ_true <= c_max + TOL)
        println("  FAIL: truth containment C: $θ_true not in [$c_min, $c_max]")
        global all_pass = false
    end
    if !(a_min - TOL <= c_min + TOL && c_max - TOL <= a_max + TOL)
        println("  FAIL: nesting A ⊇ C: A=[$a_min,$a_max] C=[$c_min,$c_max]")
        global all_pass = false
    end
    if b_min < c_min - TOL || b_max > c_max + TOL
        println("  NOTE: B wider than C (expected, SD relaxation): B=[$b_min,$b_max] C=[$c_min,$c_max]")
    end
end

println("=" ^ 100)

# ===========================================================================
# 3. Also test with known_battery=true
# ===========================================================================

println("\n" * "=" ^ 100)
println("Variant comparison (h=$h, known battery)")
println("=" ^ 100)
@printf("%-12s %4s %12s %12s %12s %12s %12s %12s %12s %12s\n",
    "Param", "t", "min_A", "max_A", "min_B", "max_B", "min_C", "max_C", "true", "in?")
println("-" ^ 100)

kb = (known_battery=true, Q_b_val=Q_b_true, P_b_val=P_b_true, SoC0_val=SoC0_true)

for t in 1:h
    a_min, _ = solve_inverse_feasibility(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=:net_load, objective_t=t, sense=:min, kb...)
    a_max, _ = solve_inverse_feasibility(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=:net_load, objective_t=t, sense=:max, kb...)
    b_min, _ = solve_inverse_optimality(fr;
        objective_var=:net_load, objective_t=t, sense=:min, known_battery=true)
    b_max, _ = solve_inverse_optimality(fr;
        objective_var=:net_load, objective_t=t, sense=:max, known_battery=true)
    c_min, _ = solve_inverse_optimality_kkt(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=:net_load, objective_t=t, sense=:min, kb...)
    c_max, _ = solve_inverse_optimality_kkt(fr.z_star, h, η_ch, η_dis, pb, ps;
        objective_var=:net_load, objective_t=t, sense=:max, kb...)

    θ_true = nl_true[t]
    in_c = c_min - TOL <= θ_true <= c_max + TOL
    @printf("%-12s %4d %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f   C:%s\n",
        "net_load", t, a_min, a_max, b_min, b_max, c_min, c_max, θ_true, in_c)

    if !in_c
        println("  FAIL: truth containment C: $θ_true not in [$c_min, $c_max]")
        global all_pass = false
    end
    if !(a_min - TOL <= c_min + TOL && c_max - TOL <= a_max + TOL)
        println("  FAIL: nesting A ⊇ C: A=[$a_min,$a_max] C=[$c_min,$c_max]")
        global all_pass = false
    end
    if b_min < c_min - TOL || b_max > c_max + TOL
        println("  NOTE: B wider than C (expected, SD relaxation): B=[$b_min,$b_max] C=[$c_min,$c_max]")
    end
end

println("=" ^ 100)

# ===========================================================================
# 4. Summary
# ===========================================================================

println("\n" * "=" ^ 60)
if all_pass
    println("Overall: PASS")
else
    println("Overall: FAIL")
    exit(1)
end
