using DataFrames, DataFramesMeta, CSV, HTTP, GLM, Random, LinearAlgebra, Statistics, Optim, Test

include("create_grids.jl")

function load_and_preprocess_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = @transform(df, :bus_id = 1:size(df,1))
    return df
end

function reshape_data(df)
    dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20, :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(Int, size(df,1))))
    select!(dfy_long, Not(:variable))

    dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10, :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(Int, size(df,1))))
    select!(dfx_long, Not(:variable))

    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    sort!(df_long, [:bus_id,:time])
    
    return df_long
end

function estimate_flexible_logit(df_long)
    formula = @formula(Y ~ (Odometer + Odometer^2 + RouteUsage + RouteUsage^2 + Branded + time + time^2)^7)
    model = glm(formula, df_long, Binomial(), LogitLink())
    return model
end

function compute_future_values(flexible_logit_model, xval, zval, xbin, zbin, T, β)
    println("Starting compute_future_values")
    println("Input sizes: xval: $(size(xval)), zval: $(size(zval)), xbin: $xbin, zbin: $zbin, T: $T")
    
    state_space = DataFrame(
        Odometer = repeat(xval, outer=zbin),
        RouteUsage = repeat(zval, inner=xbin),
        Branded = zeros(Int, xbin * zbin),
        time = zeros(Int, xbin * zbin)
    )

    println("state_space size: $(size(state_space))")

    FV = zeros(xbin * zbin, 2, T + 1)

    for t in 2:T
        for b in 0:1
            state_space.time .= t
            state_space.Branded .= b
            p0 = 1 .- predict(flexible_logit_model, state_space)
            FV[:, b+1, t] = -β * log.(p0)
        end
    end

    println("FV size: $(size(FV))")
    println("FV range: $(extrema(FV))")
    println("FV NaNs: $(sum(isnan.(FV)))")
    println("FV Infs: $(sum(isinf.(FV)))")

    return FV
end

function estimate_structural_parameters(df_long, FV, xbin, zbin, T, zval)
    println("Starting estimate_structural_parameters")
    println("Input sizes: df_long: $(size(df_long)), FV: $(size(FV)), xbin: $xbin, zbin: $zbin, T: $T, zval: $zval")
    
    FV_reshaped = zeros(nrow(df_long))
    for i in 1:nrow(df_long)
        odometer = df_long.Odometer[i]
        route_usage = df_long.RouteUsage[i]
        time = df_long.time[i]
        branded = df_long.Branded[i]

        println("Row $i: Odometer = $odometer ($(typeof(odometer))), RouteUsage = $route_usage ($(typeof(route_usage))), time = $time ($(typeof(time))), Branded = $branded ($(typeof(branded)))")

        x_index = min(max(round(Int, odometer / 5000) + 1, 1), xbin)
        z_index = argmin(abs.(zval .- route_usage))
        b = Int(branded) + 1

        println("  Calculated indices: x_index = $x_index ($(typeof(x_index))), z_index = $z_index ($(typeof(z_index))), b = $b ($(typeof(b)))")

        if !(1 <= x_index <= xbin)
            error("Invalid x_index: $x_index for Odometer: $odometer")
        end
        if !(1 <= z_index <= zbin)
            error("Invalid z_index: $z_index for RouteUsage: $route_usage")
        end
        if !(1 <= time <= T)
            error("Invalid time: $time")
        end
        if !(b in [1, 2])
            error("Invalid b: $b for Branded: $branded")
        end

        state_index = (z_index - 1) * xbin + x_index
        
        println("  state_index = $state_index ($(typeof(state_index)))")
        
        if !(1 <= state_index <= size(FV, 1))
            error("Invalid state_index: $state_index. FV size: $(size(FV))")
        end

        try
            FV_reshaped[i] = FV[state_index, b, time]
        catch e
            println("Error accessing FV: state_index = $state_index, b = $b, time = $time")
            println("FV size: $(size(FV))")
            rethrow(e)
        end
    end

    df_long.fv = FV_reshaped
    
    println("Fitting GLM model")
    model = glm(@formula(Y ~ Odometer + Branded), 
                df_long, Binomial(), LogitLink(),
                offset = df_long.fv)
    return model
end

@views function likebus(θ, d)
    FV = zeros(d.zbin*d.xbin, 2, d.T+1)
    
    for t=d.T:-1:1
        for b=0:1
            for z=1:d.zbin
                for x=1:d.xbin
                    row = x + (z-1)*d.xbin
                    v1  = θ[1] + θ[2]*d.xval[x] + θ[3]*b + d.xtran[row,           :]⋅FV[(z-1)*d.xbin+1:z*d.xbin,b+1,t+1]
                    v0  =                                  d.xtran[1+(z-1)*d.xbin,:]⋅FV[(z-1)*d.xbin+1:z*d.xbin,b+1,t+1]
                    FV[row,b+1,t] = d.β*log(exp(v1) + exp(v0))
                end
            end
        end
    end 

    like = 0
    for i=1:d.N
        row0 = (d.Zstate[i]-1)*d.xbin+1
        for t=1:d.T
            row1  = d.Xstate[i,t] + (d.Zstate[i]-1)*d.xbin
            v1    = θ[1] + θ[2]*d.X[i,t] + θ[3]*d.B[i] + (d.xtran[row1,:].-d.xtran[row0,:])⋅FV[row0:row0+d.xbin-1,d.B[i]+1,t+1]
            dem   = 1 + exp(v1)
            like -= ((d.Y[i,t]==1)*v1) - log(dem)
        end
    end
    return like
end

function estimate_dynamic_model(data_parms)
    θ_start = rand(3)
    θ_true  = [2; -.15; 1]
    
    println("Timing (twice) evaluation of the likelihood function")
    @time likebus(θ_start, data_parms)
    @time likebus(θ_start, data_parms)
    
    θ̂_optim = optimize(a -> likebus(a, data_parms), θ_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
    θ̂_ddc = θ̂_optim.minimizer
    
    return θ̂_ddc
end

function run_analysis()
    β = 0.9

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = load_and_preprocess_data(url)
    df_long = reshape_data(df)
    
    println("df_long summary:")
    println(describe(df_long))
    
    flexible_logit_model = estimate_flexible_logit(df_long)
    println("Flexible Logit Model Results:")
    println(flexible_logit_model)
    
    Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
    X = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    Z = Vector(df[:,:RouteUsage])
    B = Vector(df[:,:Branded])
    N, T = size(Y)
    Xstate = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    Zstate = Vector(df[:,:Zst])
    
    zval, zbin, xval, xbin, xtran = create_grids()
    
    println("Grid information:")
    println("zval: ", zval)
    println("zbin: ", zbin)
    println("xval: ", xval)
    println("xbin: ", xbin)
    
    FV = compute_future_values(flexible_logit_model, xval, zval, xbin, zbin, T, β)
    
    println("FV shape: ", size(FV))
    println("FV range: ", extrema(FV))
    println("FV NaNs: ", sum(isnan.(FV)))
    println("FV Infs: ", sum(isinf.(FV)))
    
    structural_model = estimate_structural_parameters(df_long, FV, xbin, zbin, T, zval)
    println("Structural Model Results:")
    println(structural_model)
    
    data_parms = (β = β, Y = Y, B = B, N = N, T = T, X = X, Z = Z, Zstate = Zstate, Xstate = Xstate,
                  xtran = xtran, zbin = zbin, xbin = xbin, xval = xval)
    
    θ̂_ddc = estimate_dynamic_model(data_parms)
    println("Dynamic Model Results:")
    println(θ̂_ddc)
end

function run_tests()
    @testset "Bus Engine Replacement Model Tests" begin
        @testset "Data Preparation" begin
            url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
            df = load_and_preprocess_data(url)
            @test size(df, 1) > 0
            @test :bus_id in names(df)
            
            df_long = reshape_data(df)
            @test size(df_long, 1) == size(df, 1) * 20
            @test all([:Y, :Odometer, :time, :RouteUsage, :Branded] .∈ Ref(names(df_long)))
        end

        @testset "Flexible Logit Model" begin
            df_long = DataFrame(Y = rand(0:1, 100), Odometer = rand(100:1000, 100),
                                RouteUsage = rand(1:5, 100), Branded = rand(0:1, 100),
                                time = rand(1:20, 100))
            model = estimate_flexible_logit(df_long)
            @test isa(model, GLM.GeneralizedLinearModel)
            @test length(coef(model)) > 3  # Should have many interaction terms
        end

        @testset "Future Value Computation" begin
            df_long = DataFrame(Y = rand(0:1, 100), Odometer = rand(100:1000, 100),
                                RouteUsage = rand(1:5, 100), Branded = rand(0:1, 100),
                                time = rand(1:20, 100))
            dummy_model = estimate_flexible_logit(df_long)
            xval = collect(100:100:1000)
            zval = collect(1:5)
            xbin = length(xval)
            zbin = length(zval)
            T = 20
            β = 0.9

            FV = compute_future_values(dummy_model, xval, zval, xbin, zbin, T, β)
            @test size(FV) == (xbin * zbin, 2, T + 1)
            @test all(FV .<= 0)  # Future values should be non-positive
        end

        @testset "Structural Parameter Estimation" begin
            df_long = DataFrame(Y = rand(0:1, 100), Odometer = rand(100:1000, 100),
                                RouteUsage = rand(1:5, 100), Branded = rand(0:1, 100),
                                time = rand(1:20, 100))
            xval = collect(100:100:1000)
            zval = collect(1:5)
            xbin = length(xval)
            zbin = length(zval)
            T = 20
            FV = rand(xbin * zbin, 2, T + 1)
            model = estimate_structural_parameters(df_long, FV, xbin, zbin, T, zval)
            @test isa(model, GLM.GeneralizedLinearModel)
            @test length(coef(model)) == 3  # Intercept, Odometer, Branded
        end

        @testset "Dynamic Model Likelihood" begin
            N, T = 10, 20
            data_parms = (
                β = 0.9,
                Y = rand(0:1, N, T),
                B = rand(0:1, N),
                N = N,
                T = T,
                X = rand(100:1000, N, T),
                Z = rand(1:5, N),
                Zstate = rand(1:5, N),
                Xstate = rand(1:10, N, T),
                xtran = rand(10, 10),
                zbin = 5,
                xbin = 10,
                xval = collect(100:100:1000)
            )
            
            θ_test = [0.5, -0.01, 1.0]
            
            @test isa(likebus(θ_test, data_parms), Number)
        end
    end
end

@time run_analysis()
run_tests()