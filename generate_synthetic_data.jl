using CSV
using DataFrames
using Dates
using Random
using Statistics

using CSV: Schema

const NUM_BUILDINGS = 30
const NUM_DAYS = 7
const STEPS_PER_DAY = 96
const NUM_STEPS = NUM_DAYS * STEPS_PER_DAY
const PRED_HORIZON = 96

const OUT_DIR = "synthetic_data"
const OUT_DATA = joinpath(OUT_DIR, "data")
const OUT_CLEAN = joinpath(OUT_DIR, "cleaned_data")

struct Archetype
    name::String
    cons_func::Function
    gen_func::Function
    gen_type::Symbol
    capacity::Float64
    power::Float64
    charge_eff::Float64
    discharge_eff::Float64
end

function gaussian_peak(t::Real; center::Real, width::Real, amplitude::Real)
    return amplitude * exp(-((t - center)^2) / (2 * width^2))
end

function solar_bell(t::Real; amplitude::Real, sunrise::Real=28.0, sunset::Real=68.0)
    if t < sunrise || t > sunset
        return 0.0
    end
    day_frac = (t - sunrise) / (sunset - sunrise)
    return amplitude * sin(pi * day_frac)
end

function coastal_wind_envelope(t::Real; rated::Real=1.0)
    step = mod(t, STEPS_PER_DAY)
    env = 0.2 + 0.8 * (0.5 + 0.5 * cos(2 * pi * (step - 12) / STEPS_PER_DAY))
    return rated * env
end

function hilltop_wind_envelope(t::Real; rated::Real=1.0)
    step = mod(t, STEPS_PER_DAY)
    env = 0.15 + 0.85 * (0.5 + 0.5 * sin(2 * pi * (step - 44) / STEPS_PER_DAY))
    return rated * env
end

function flat_load(t::Real; base::Real)
    return base
end

function office_load(t::Real; base::Real, peak::Real)
    step = mod(t, STEPS_PER_DAY)
    weekday = div(Int(step), STEPS_PER_DAY) == 0
    morning = gaussian_peak(step, center=40.0, width=4.0, amplitude=peak)
    afternoon = gaussian_peak(step, center=56.0, width=6.0, amplitude=peak * 0.7)
    return base + morning + afternoon
end

function residential_load(t::Real; base::Real, am_peak::Real, pm_peak::Real)
    step = mod(t, STEPS_PER_DAY)
    morning = gaussian_peak(step, center=32.0, width=3.0, amplitude=am_peak)
    evening = gaussian_peak(step, center=76.0, width=4.0, amplitude=pm_peak)
    return base + morning + evening
end

function industrial_load(t::Real; base::Real)
    step = mod(t, STEPS_PER_DAY)
    weekday_shift = 0.9 + 0.1 * sin(2 * pi * step / STEPS_PER_DAY)
    return base * weekday_shift
end

function datacenter_load(t::Real; base::Real)
    step = mod(t, STEPS_PER_DAY)
    day_cycle = 0.85 + 0.15 * sin(2 * pi * (step - 12) / STEPS_PER_DAY)
    return base * day_cycle
end

function warehouse_load(t::Real; base::Real, peak::Real)
    step = mod(t, STEPS_PER_DAY)
    active = gaussian_peak(step, center=40.0, width=12.0, amplitude=peak)
    return base + active
end

function school_load(t::Real; base::Real, peak::Real)
    step = mod(t, STEPS_PER_DAY)
    day_index = div(Int(t), STEPS_PER_DAY)
    is_weekend = day_index >= 5
    if is_weekend
        return base
    end
    morning = gaussian_peak(step, center=40.0, width=5.0, amplitude=peak)
    return base + morning
end

function restaurant_load(t::Real; base::Real, lunch_peak::Real, dinner_peak::Real)
    step = mod(t, STEPS_PER_DAY)
    lunch = gaussian_peak(step, center=50.0, width=3.0, amplitude=lunch_peak)
    dinner = gaussian_peak(step, center=76.0, width=4.0, amplitude=dinner_peak)
    return base + lunch + dinner
end

const ARCHETYPES = [
    Archetype("solar_farm",
        (t, s) -> flat_load(t; base=s),
        (t, s) -> solar_bell(t; amplitude=s),
        :solar, 500.0, 125.0, 0.97, 0.97),
    Archetype("office",
        (t, s) -> office_load(t; base=s[1], peak=s[2]),
        (t, s) -> solar_bell(t; amplitude=s[1]),
        :solar, 100.0, 25.0, 0.92, 0.92),
    Archetype("residential",
        (t, s) -> residential_load(t; base=s[1], am_peak=s[2], pm_peak=s[3]),
        (t, s) -> solar_bell(t; amplitude=s[1]),
        :solar, 20.0, 5.0, 0.90, 0.90),
    Archetype("industrial",
        (t, s) -> industrial_load(t; base=s),
        (t, s) -> 0.0,
        :none, 300.0, 75.0, 0.95, 0.95),
    Archetype("warehouse_pv",
        (t, s) -> warehouse_load(t; base=s[1], peak=s[2]),
        (t, s) -> solar_bell(t; amplitude=s[1]),
        :solar, 150.0, 37.5, 0.93, 0.93),
    Archetype("data_center",
        (t, s) -> datacenter_load(t; base=s),
        (t, s) -> 0.0,
        :none, 200.0, 50.0, 0.95, 0.95),
    Archetype("school",
        (t, s) -> school_load(t; base=s[1], peak=s[2]),
        (t, s) -> solar_bell(t; amplitude=s[1]),
        :solar, 50.0, 12.5, 0.90, 0.90),
    Archetype("restaurant",
        (t, s) -> restaurant_load(t; base=s[1], lunch_peak=s[2], dinner_peak=s[3]),
        (t, s) -> 0.0,
        :none, 15.0, 3.75, 0.88, 0.88),
    Archetype("coastal_wind",
        (t, s) -> flat_load(t; base=s),
        (t, s) -> coastal_wind_envelope(t; rated=s),
        :wind, 400.0, 100.0, 0.96, 0.96),
    Archetype("hilltop_wind",
        (t, s) -> flat_load(t; base=s),
        (t, s) -> hilltop_wind_envelope(t; rated=s),
        :wind, 350.0, 87.5, 0.95, 0.95),
]

const NUM_ARCHETYPES = length(ARCHETYPES)
const INSTANCES_PER_ARCHETYPE = div(NUM_BUILDINGS, NUM_ARCHETYPES)

function cons_scale_params(arch::Archetype, instance::Int, rng::AbstractRNG)
    variation = 1.0 + 0.2 * (instance - 2) / max(INSTANCES_PER_ARCHETYPE - 1, 1)
    name = arch.name
    if name == "solar_farm"
        return 5.0 * variation
    elseif name == "office"
        return [20.0 * variation, 60.0 * variation]
    elseif name == "residential"
        return [3.0 * variation, 8.0 * variation, 12.0 * variation]
    elseif name == "industrial"
        return 150.0 * variation
    elseif name == "warehouse_pv"
        return [10.0 * variation, 40.0 * variation]
    elseif name == "data_center"
        return 200.0 * variation
    elseif name == "school"
        return [5.0 * variation, 30.0 * variation]
    elseif name == "restaurant"
        return [3.0 * variation, 15.0 * variation, 20.0 * variation]
    elseif name == "coastal_wind"
        return 3.0 * variation
    elseif name == "hilltop_wind"
        return 3.0 * variation
    end
end

function gen_scale_params(arch::Archetype, instance::Int, rng::AbstractRNG)
    variation = 1.0 + 0.2 * (instance - 2) / max(INSTANCES_PER_ARCHETYPE - 1, 1)
    name = arch.name
    if name == "solar_farm"
        return 300.0 * variation
    elseif name == "office"
        return 80.0 * variation
    elseif name == "residential"
        return 10.0 * variation
    elseif name == "industrial"
        return 0.0
    elseif name == "warehouse_pv"
        return 200.0 * variation
    elseif name == "data_center"
        return 0.0
    elseif name == "school"
        return 40.0 * variation
    elseif name == "restaurant"
        return 0.0
    elseif name == "coastal_wind"
        return 200.0 * variation
    elseif name == "hilltop_wind"
        return 180.0 * variation
    end
end

function battery_params(arch::Archetype, instance::Int, rng::AbstractRNG)
    variation = 1.0 + 0.1 * (instance - 2) / max(INSTANCES_PER_ARCHETYPE - 1, 1)
    cap = arch.capacity * variation
    pwr = arch.power * variation
    ce = min(arch.charge_eff + 0.02 * randn(rng), 0.99)
    de = min(arch.discharge_eff + 0.02 * randn(rng), 0.99)
    return cap, pwr, max(ce, 0.80), max(de, 0.80)
end

function generate_wind_profile(arch::Archetype, gen_scale::Float64, rng::AbstractRNG)
    profile = zeros(NUM_STEPS)
    noise = zeros(NUM_STEPS)
    if arch.name == "coastal_wind"
        phi, sigma = 0.8, 0.30
    else
        phi, sigma = 0.7, 0.50
    end
    noise[1] = sigma * randn(rng)
    for t in 2:NUM_STEPS
        noise[t] = phi * noise[t-1] + sigma * randn(rng)
    end
    for t in 1:NUM_STEPS
        env_val = arch.gen_func(t, gen_scale)
        wind_val = env_val * (1.0 + noise[t])
        profile[t] = max(wind_val, 0.0)
    end
    return profile
end

function generate_solar_profile(arch::Archetype, gen_scale::Float64, rng::AbstractRNG)
    profile = zeros(NUM_STEPS)
    for t in 1:NUM_STEPS
        day_index = div(t - 1, STEPS_PER_DAY)
        daily_var = 0.85 + 0.30 * randn(rng)
        daily_var = clamp(daily_var, 0.5, 1.3)
        cloud = 1.0
        if rand(rng) < 0.15
            cloud = 0.6 + 0.3 * rand(rng)
        end
        val = arch.gen_func(t, gen_scale) * daily_var * cloud
        profile[t] = max(val, 0.0)
    end
    return profile
end

function generate_load_profile(arch::Archetype, cons_scale, rng::AbstractRNG)
    profile = zeros(NUM_STEPS)
    for t in 1:NUM_STEPS
        val = arch.cons_func(t, cons_scale)
        val *= (1.0 + 0.05 * randn(rng))
        profile[t] = max(val, 0.0)
    end
    return profile
end

function generate_predictions(actual::Vector{Float64}, sigma::Float64, rng::AbstractRNG)
    positives = filter(x -> x > 0, actual)
    mean_val = isempty(positives) ? 1.0 : mean(positives)
    if !isfinite(mean_val) || mean_val == 0
        mean_val = 1.0
    end
    pred = zeros(NUM_STEPS, PRED_HORIZON)
    for t in 1:NUM_STEPS
        for j in 1:PRED_HORIZON
            idx = mod(t + j - 2, NUM_STEPS) + 1
            noise = sigma * mean_val * randn(rng)
            pred[t, j] = max(actual[idx] + noise, 0.0)
        end
    end
    return pred
end

function generate_prices(rng::AbstractRNG)
    buy = zeros(STEPS_PER_DAY)
    sell = zeros(STEPS_PER_DAY)
    for t in 1:STEPS_PER_DAY
        if t <= 24
            buy[t] = 0.08
            sell[t] = 0.03
        elseif t <= 36
            buy[t] = 0.30
            sell[t] = 0.12
        elseif t <= 64
            buy[t] = 0.15
            sell[t] = 0.06
        elseif t <= 80
            buy[t] = 0.30
            sell[t] = 0.12
        else
            buy[t] = 0.15
            sell[t] = 0.06
        end
    end
    for day in 1:NUM_DAYS
        if day > 1
            n_spikes = 1 + rand(rng, 0:2)
            for _ in 1:n_spikes
                t = rand(rng, 1:STEPS_PER_DAY)
                buy[t] = 0.50
                sell[t] = 0.20
            end
        end
    end
    times = Vector{String}(undef, STEPS_PER_DAY)
    for t in 1:STEPS_PER_DAY
        h = div(t - 1, 4)
        m = (mod(t - 1, 4)) * 15
        times[t] = string(lpad(string(h), 2, '0'), ":", lpad(string(m), 2, '0'), ":00")
    end
    df = DataFrame(timestamp=times, buy=buy, sell=sell)
    return df
end

function generate_metadata(buildings_info::Vector{NamedTuple})
    df = DataFrame(
        site_id = Int[],
        max_load = Float64[],
        capacity = Float64[],
        power = Float64[],
        charge_efficiency = Float64[],
        discharge_efficiency = Float64[],
    )
    for info in buildings_info
        push!(df, (
            info.id,
            info.max_load,
            info.capacity,
            info.power,
            info.charge_eff,
            info.discharge_eff,
        ))
    end
    return df
end

function generate_datetimes()
    start = DateTime(2024, 1, 1)
    dts = Vector{String}(undef, NUM_STEPS)
    for t in 1:NUM_STEPS
        dt = start + Dates.Minute(15 * (t - 1))
        dts[t] = string(Dates.format(dt, "mm-ddTHH:MM:SS"), "+00:00")
    end
    return dts
end

function generate_building_csv(building_id::Int, datetimes::Vector{String},
        load_actual::Vector{Float64}, gen_actual::Vector{Float64},
        load_pred::Matrix{Float64}, gen_pred::Matrix{Float64})::DataFrame
    df = DataFrame(DateTime = datetimes)
    df.site_id_mean = fill(building_id, NUM_STEPS)
    df.actual_consumption_mean = load_actual
    df.actual_pv_mean = gen_actual
    for j in 0:PRED_HORIZON-1
        col_name = string("load_", lpad(string(j), 2, '0'), "_mean")
        df[!, col_name] = load_pred[:, j+1]
    end
    for j in 0:PRED_HORIZON-1
        col_name = string("pv_", lpad(string(j), 2, '0'), "_mean")
        df[!, col_name] = gen_pred[:, j+1]
    end
    return df
end

function main()
    rng = MersenneTwister(42)
    mkpath(OUT_DATA)
    mkpath(OUT_CLEAN)

    datetimes = generate_datetimes()
    price_df = generate_prices(rng)
    CSV.write(joinpath(OUT_DATA, "edf_prices.csv"), price_df)
    println("Wrote edf_prices.csv ($(nrow(price_df)) rows)")

    buildings_info = Vector{NamedTuple}()
    bid = 0
    for (arch_idx, arch) in enumerate(ARCHETYPES)
        for inst in 1:INSTANCES_PER_ARCHETYPE
            bid += 1
            cons_scale = cons_scale_params(arch, inst, rng)
            gen_scale = gen_scale_params(arch, inst, rng)
            cap, pwr, ce, de = battery_params(arch, inst, rng)

            load_actual = generate_load_profile(arch, cons_scale, rng)
            if arch.gen_type == :wind
                gen_actual = generate_wind_profile(arch, gen_scale, rng)
                pred_sigma = 0.20
            elseif arch.gen_type == :solar
                gen_actual = generate_solar_profile(arch, gen_scale, rng)
                pred_sigma = 0.08
            else
                gen_actual = zeros(NUM_STEPS)
                pred_sigma = 0.08
            end

            load_pred = generate_predictions(load_actual, 0.08, rng)
            gen_pred = generate_predictions(gen_actual, pred_sigma, rng)

            max_load = maximum(load_actual)

            push!(buildings_info, (
                id = bid,
                max_load = max_load,
                capacity = cap,
                power = pwr,
                charge_eff = ce,
                discharge_eff = de,
            ))

            df = generate_building_csv(bid, datetimes, load_actual, gen_actual, load_pred, gen_pred)
            filename = joinpath(OUT_CLEAN, string(bid, ".csv"))
            CSV.write(filename, df)
            println("Wrote $filename ($(nrow(df)) rows, $(ncol(df)) cols) [$(arch.name) #$inst]")
        end
    end

    meta_df = generate_metadata(buildings_info)
    CSV.write(joinpath(OUT_DATA, "metadata.csv"), meta_df)
    println("Wrote metadata.csv ($(nrow(meta_df)) buildings)")

    println("\n=== Summary ===")
    for info in buildings_info
        arch = ARCHETYPES[div(info.id - 1, INSTANCES_PER_ARCHETYPE) + 1]
        println("  Bldg $(info.id): $(arch.name)  cap=$(round(info.capacity, digits=1))  power=$(round(info.power, digits=1))  eff=($(round(info.charge_eff, digits=2)),$(round(info.discharge_eff, digits=2)))  max_load=$(round(info.max_load, digits=1))")
    end

    println("\nDone. Synthetic data written to $OUT_DIR/")
end

main()
