abstract type Player end

struct Building <: Player
	loc::Tuple
	cons::Vector{Float64}
	prod::Vector{Float64}
	max_storage::Float16
	id::Int
end

Base.show(io::IO, b::Building) = print(io, b.id)

location(b::Building) = b.loc
consumption(b::Building) = b.cons
production(b::Building) = b.prod