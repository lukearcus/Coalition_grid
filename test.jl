
abstract type Player end

struct Building <: Player
	loc::Tuple
	cons::Vector{Float64}
	prod::Vector{Float64}
	max_storage::Float16
end

location(b::Building) = b.loc
consumption(b::Building) = b.cons
production(b::Building) = b.prod

#rand_num = 
#println(rand_num)
buildings = [Building((rand(Float64, 1)[1], rand(Float64, 1)[1]), rand(Float64, 24), rand(Float64, 24), rand(Float16, 1)[1] ) for i = 1:10]

#build_1 = Building((0.0,0.0),[1,2,3],[3,2,1],2)

println("Building 1 is in location ", buildings[1].loc)
println("Building 1 requires energy ", consumption(buildings[1]))
