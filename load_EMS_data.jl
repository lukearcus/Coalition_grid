using CSV
using DataFrames

include("Buildings.jl")

max_builds = 70


#println(meta)

function load_from_CSV(num_builds::Int)
    meta = CSV.read("data/metadata.csv", DataFrame)
    if num_builds < 1
        num_builds = 1
    elseif num_builds > max_builds
        num_builds = max_builds
    end
    builds = Vector{Building}()

    for i in 1:num_builds
        filename = "data/"*string(i)*".csv"
        file = CSV.read(filename, DataFrame, delim=";")
        #Currently, use only predicted data from day 1, site loc is randomly assigned,
        # battery power and charge efficiencies stored but not implemented
        build = Building((rand(Float64, 1)[1], rand(Float64, 1)[1]), collect(file[1, Cols(x -> startswith(x, "load_"))]), collect(file[1, Cols(x -> startswith(x, "pv_"))]), meta[i,:capacity], meta[i,:power], meta[i,:charge_efficiency],meta[i,:discharge_efficiency],i)
        push!(builds, build)
        #println(collect(file[1, Cols(x -> startswith(x, "load_"))]))
    end

    price_file = CSV.read("data/edf_prices.csv", DataFrame)
    buy = collect(price_file[1:96,:buy])
    sell = collect(price_file[1:96,:sell])

    return builds, buy, sell
end