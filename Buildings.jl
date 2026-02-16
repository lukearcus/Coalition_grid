abstract type Player end

struct Building <: Player
	loc::Tuple
	cons::Vector{Float64}
	prod::Vector{Float64}
	max_storage::Float16
	storage_max_flow::Float16
	charge_eff::Float16
	discharge_eff::Float16
	id::Int
end

Base.show(io::IO, b::Building) = print(io, b.id)

location(b::Building) = b.loc
consumption(b::Building) = b.cons
production(b::Building) = b.prod
max_store(b::Building) = b.max_storage
max_flow(b::Building) = b.storage_max_flow
charge_eff(b::Building) = b.charge_eff
discharge_eff(b::Building) = b.discharge_eff

struct MPC_Building <: Player
	loc::Tuple
	pred_cons::Matrix{Float64}
	pred_prod::Matrix{Float64}
	act_cons::Vector{Float64}
	act_prod::Vector{Float64}
	max_storage::Float16
	storage_max_flow::Float16
	charge_eff::Float16
	discharge_eff::Float16
	id::Int
	SoC::Vector{Float64}
end

Base.show(io::IO, b::MPC_Building) = print(io, b.id)

location(b::MPC_Building) = b.loc
pred_consumption(b::MPC_Building,k::Int) = b.pred_cons[k,:]
pred_production(b::MPC_Building,k::Int) = b.pred_prod[k,:]
max_store(b::MPC_Building) = b.max_storage
max_flow(b::MPC_Building) = b.storage_max_flow
charge_eff(b::MPC_Building) = b.charge_eff
discharge_eff(b::MPC_Building) = b.discharge_eff



