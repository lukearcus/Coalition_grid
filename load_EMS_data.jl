try
    using CSV
catch
    using CSV
end
using DataFrames
using Dates
using Statistics

include("Buildings.jl")

max_builds = 70


#println(meta)

function MPC_load_from_CSV(num_builds::Int,num_steps::Int)
    meta = CSV.read("data/metadata.csv", DataFrame)
    if num_builds < 1
        num_builds = 1
    elseif num_builds > max_builds
        num_builds = max_builds
    end
    if num_steps < 1
        num_steps = 1
    end
    builds = Vector{MPC_Building}()

    for i in 1:num_builds
        filename = "cleaned_data/"*string(i)*".csv"
        file = CSV.read(filename, DataFrame, delim=",")
        if num_steps > nrow(file)
            num_steps = nrow(file)
        end
        #Currently, use only predicted data from day 1, site loc is randomly assigned,
        # battery power and charge efficiencies stored but not implemented
        build = MPC_Building((rand(Float64, 1)[1], rand(Float64, 1)[1]), Matrix(file[1:num_steps, Cols(x -> startswith(x, "load_"))]), Matrix(file[1:num_steps, Cols(x -> startswith(x, "pv_"))]), CSV.File(filename; select=[3]).actual_consumption_mean[1:num_steps], CSV.File(filename; select=[4]).actual_pv_mean[1:num_steps], meta[i,:capacity], meta[i,:power], meta[i,:charge_efficiency],meta[i,:discharge_efficiency],i,zeros(num_steps))
        push!(builds, build)
        #println(collect(file[1, Cols(x -> startswith(x, "load_"))]))
    end
    filename = "cleaned_data/"*string(1)*".csv"
    file = CSV.read(filename, DataFrame, delim=",")
    start = split(split(file[1,:DateTime],"+")[1], "T")[2]

    price_file = CSV.read("data/edf_prices.csv", DataFrame)
    start_ind = 0
    for i in 1:nrow(price_file)
        if price_file[i,1] == Time(start)
            start_ind = i
            break
        end
    end
    # println(start_ind)
    buy = collect(price_file[start_ind:start_ind+95,:buy])
    sell = collect(price_file[start_ind:start_ind+95,:sell])

    return builds, buy, sell
end
function load_from_CSV(num_builds::Int,num_steps::Int)
    meta = CSV.read("data/metadata.csv", DataFrame)
    if num_builds < 1
        num_builds = 1
    elseif num_builds > max_builds
        num_builds = max_builds
    end
    if num_steps < 1
        num_steps = 1
    end
    builds = Vector{Building}()

    for i in 1:num_builds
        filename = "cleaned_data/"*string(i)*".csv"
        file = CSV.read(filename, DataFrame, delim=",")
        if num_steps > nrow(file)
            num_steps = nrow(file)
        end
        #Currently, use only predicted data from day 1, site loc is randomly assigned,
        # battery power and charge efficiencies stored but not implemented
        build = Building((rand(Float64, 1)[1], rand(Float64, 1)[1]), CSV.File(filename; select=[3]).actual_consumption_mean[1:num_steps], CSV.File(filename; select=[4]).actual_pv_mean[1:num_steps], meta[i,:capacity], meta[i,:power], meta[i,:charge_efficiency],meta[i,:discharge_efficiency],i)
        push!(builds, build)
        #println(collect(file[1, Cols(x -> startswith(x, "load_"))]))
    end
    filename = "cleaned_data/"*string(1)*".csv"
    file = CSV.read(filename, DataFrame, delim=",")
    start = split(split(file[1,:DateTime],"+")[1], "T")[2]

    price_file = CSV.read("data/edf_prices.csv", DataFrame)
    start_ind = 0
    for i in 1:nrow(price_file)
        if price_file[i,1] == Time(start)
            start_ind = i
            break
        end
    end
    # println(start_ind)
    buy = collect(price_file[start_ind:start_ind+95,:buy])
    sell = collect(price_file[start_ind:start_ind+95,:sell])

    return builds, buy, sell
end

function clean_data(start_t::Int,num_steps::Int)
    @eval CSV.Parsers import Base.Ryu: writeshortest
    start = DateTime(2012)
    period = [start]
    current = start
    for t in 1:35135
        current += Dates.Minute(15)
        period = vcat(period,current)
    end
    period = [string(string(p)[6:19] , "+00:00") for p in period]
    for i in 1:70
        filename = "data/"*string(i)*".csv"
        master = DataFrame(DateTime = period)
        file = CSV.read(filename, DataFrame, delim=";")
        file = transform(file, :timestamp => ByRow(x->split(x,'-')) => [:Year, :month, :dayTime])
        file.DateTime = string.(file.month,"-", file.dayTime)
        select!(file, Not(:Year))
        select!(file, Not(:month))
        select!(file, Not(:dayTime))
        select!(file, Not(:timestamp))
        select!(file, Not(:period_id))
        grouped = groupby(file, :DateTime)
        combined = combine(grouped, Not(:DateTime).=>mean)
        master = leftjoin(master, combined, on=:DateTime, makeunique=true)
        master.site_id_mean .= i
        sort!(master)
        master = master[start_t:start_t+num_steps-1,:]

        cleaned_filename = "cleaned_data/"*string(i)*".csv"
        CSV.write(cleaned_filename, master)
    end
end



function find_starts(num_steps::Int)

    start = DateTime(2012)
    period = [start]
    current = start
    for t in 1:35135
        current += Dates.Minute(15)
        period = vcat(period,current)
    end
    period = [string(string(p)[6:19] , "+00:00") for p in period]
    possible_starts = [i for i in 1:35136-num_steps+1]
    for i in 1:70
        filename = "data/"*string(i)*".csv"
        master = DataFrame(DateTime = period)
        file = CSV.read(filename, DataFrame, delim=";")
        file = transform(file, :timestamp => ByRow(x->split(x,'-')) => [:Year, :month, :dayTime])
        file.DateTime = string.(file.month,"-", file.dayTime)
        select!(file, Not(:Year))
        select!(file, Not(:month))
        select!(file, Not(:dayTime))
        select!(file, Not(:timestamp))
        select!(file, Not(:period_id))
        grouped = groupby(file, :DateTime)
        combined = combine(grouped, Not(:DateTime).=>mean)
        master = leftjoin(master, combined, on=:DateTime, makeunique=true)
        master.site_id_mean .= i
        sort!(master)
        for start in possible_starts
            if start != -1
                if sum(master[start:start+num_steps-1,3]) isa Missing
                    possible_starts[start] = -1
                end
            end
        end
    end
    possible_starts = [i for i in possible_starts if i != -1]
    return possible_starts
end

function find_max_start()
    lb = 96
    ub = 35136
    while true
        i = Int(round((ub+lb)/2))
        println(i)
        if length(find_starts(i)) > 0
            lb = i
        else
            ub = i
        end
        if lb + 1 == ub
            break
        end
    end
    return lb
end

function check_start_end()

    max_start = 0
    min_end = 0
    for i in 1:7
        filename = "data/"*string(i)*".csv"
        file = CSV.read(filename, DataFrame, delim=";")
        num_builds = max_builds
        #Currently, use only predicted data from day 1, site loc is randomly assigned,
        # battery power and charge efficiencies stored but not implemented
        start = DateTime.(split(file[1,:timestamp],"+")[1])
        endT = DateTime.(split(file[nrow(file),:timestamp],"+")[1])
        if max_start==0
            max_start=start
        elseif start > max_start
            max_start=start
        end
        if min_end==0
            min_end=endT
        elseif endT < min_end
            min_end = endT
        end
        #println(file[1,:timestamp])
        # build = MPC_Building((rand(Float64, 1)[1], rand(Float64, 1)[1]), Matrix(file[1:num_steps, Cols(x -> startswith(x, "load_"))]), Matrix(file[1:num_steps, Cols(x -> startswith(x, "pv_"))]), CSV.File(filename; select=[4]).actual_consumption[1:num_steps], CSV.File(filename; select=[5]).actual_pv[1:num_steps], meta[i,:capacity], meta[i,:power], meta[i,:charge_efficiency],meta[i,:discharge_efficiency],i,zeros(num_steps))
        # push!(builds, build)
        #println(collect(file[1, Cols(x -> startswith(x, "load_"))]))
    end
    println(max_start)
    println(min_end)
    # price_file = CSV.read("data/edf_prices.csv", DataFrame)
    # buy = collect(price_file[1:96,:buy])
    # sell = collect(price_file[1:96,:sell])

end