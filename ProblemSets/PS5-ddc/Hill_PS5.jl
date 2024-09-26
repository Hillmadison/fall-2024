# Problem Set 5
# Madison Hill 

using Pkg

pkgs = ["Random", "LinearAlgebra", "Statistics", "Optim", "DataFrames", "DataFramesMeta", 
        "CSV", "HTTP", "GLM", "Test", "PrecompileTools", "StaticArrays"]

# Check which packages need to be installed
to_install = filter(p -> p ∉ keys(Pkg.project().dependencies), pkgs)

# Only run Pkg.add if there are packages to install
if !isempty(to_install)
    Pkg.add(to_install)
end

# Load all packages
foreach(p -> eval(Meta.parse("using $p")), pkgs)

using Base.Threads

println("Number of threads: ", Threads.nthreads())

# Read in function to create state transitions for dynamic model
include("create_grids.jl")

@setup_workload begin
    # Precompile the functions you'll use
    @compile_workload begin
        # Add small-scale versions of your main functions here
        # This will be filled in after we define our main functions
    end
end

# Question 1: Reshaping the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Create bus id variable
df = @transform(df, :bus_id = 1:size(df,1))

# Reshape from wide to long
# First reshape the decision variable
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, :time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

# Next reshape the odometer variable
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, :time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

# Join reshaped df's back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])

# Question 2: Estimate a static version of the model
function estimate_static_model(df_long)
    model = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
    return coef(model)
end

static_coefficients = estimate_static_model(df_long)
println("Static Model Coefficients:")
println("θ₀ (Intercept): ", static_coefficients[1])
println("θ₁ (Odometer): ", static_coefficients[2])
println("θ₂ (Branded): ", static_coefficients[3])

# Question 3: Dynamic estimation
# 3a: Read in data for dynamic model
url_dynamic = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
df_dynamic = CSV.read(HTTP.get(url_dynamic).body, DataFrame)

Y = Matrix{Int}(df_dynamic[:, r"^Y"])
Odo = Matrix{Float64}(df_dynamic[:, r"^Odo"])
Xst = Matrix{Int}(df_dynamic[:, r"^Xst"])
Zst = Vector{Int}(df_dynamic.Zst)
B = Vector{Int}(df_dynamic.Branded)

# 3b: Generate state transition matrices
zval, zbin, xval, xbin, xtran = create_grids()

# 3c: Compute future value terms
function compute_future_values(θ::SVector{3,Float64}, β::Float64, T::Int, zbin::Int, xbin::Int, xval::Vector{Float64}, xtran::Matrix{Float64}, B::Vector{Int})
    FV = zeros(zbin * xbin, 2, T + 1)
    
    for t in T:-1:1
        Threads.@threads for z in 1:zbin
            for b in 0:1, x in 1:xbin
                row = x + (z-1)*xbin
                v1t = θ[1] + θ[2] * xval[x] + θ[3] * b + 
                      β * dot(view(xtran, row, :), view(FV, (z-1)*xbin+1:z*xbin, b+1, t+1))
                v0t = β * dot(view(xtran, 1+(z-1)*xbin, :), view(FV, (z-1)*xbin+1:z*xbin, b+1, t+1))
                FV[row, b+1, t] = β * log(exp(v0t) + exp(v1t))
            end
        end
    end
    
    return FV
end

# 3d: Construct log likelihood function
function log_likelihood(θ::SVector{3,Float64}, β::Float64, T::Int, zbin::Int, xbin::Int, xval::Vector{Float64}, xtran::Matrix{Float64}, Y::Matrix{Int}, Odo::Matrix{Float64}, Xst::Matrix{Int}, Zst::Vector{Int}, B::Vector{Int})
    FV = compute_future_values(θ, β, T, zbin, xbin, xval, xtran, B)
    ll = Threads.Atomic{Float64}(0.0)
    
    Threads.@threads for i in 1:size(Y, 1)
        local_ll = 0.0
        for t in 1:T
            row0 = 1 + (Zst[i]-1)*xbin
            row1 = Xst[i,t] + (Zst[i]-1)*xbin
            
            v1t_v0t = θ[1] + θ[2] * Odo[i,t] + θ[3] * B[i] + 
                      β * dot(view(xtran, row1, :) .- view(xtran, row0, :), view(FV, row0:row0+xbin-1, B[i]+1, t+1))
            
            P1 = 1 / (1 + exp(-v1t_v0t))
            local_ll += Y[i,t] * log(P1) + (1 - Y[i,t]) * log(1 - P1)
        end
        Threads.atomic_add!(ll, local_ll)
    end
    
    return -ll[]  # Return negative log-likelihood for minimization
end

# 3e-3h: Estimate dynamic model
@views @inbounds function estimate_dynamic_model(Y::Matrix{Int}, Odo::Matrix{Float64}, Xst::Matrix{Int}, Zst::Vector{Int}, B::Vector{Int}, β::Float64=0.9, T::Int=20)
    zval, zbin, xval, xbin, xtran = create_grids()
    
    function objective(θ::Vector{Float64})
        return log_likelihood(SVector{3,Float64}(θ), β, T, zbin, xbin, xval, xtran, Y, Odo, Xst, Zst, B)
    end
    
    # Use static model estimates as initial values
    initial_θ = static_coefficients
    
    result = optimize(objective, initial_θ, LBFGS(), Optim.Options(iterations=1000))
    
    return Optim.minimizer(result)
end

# Estimate the dynamic model
@time dynamic_coefficients = estimate_dynamic_model(Y, Odo, Xst, Zst, B)
println("Dynamic Model Coefficients:")
println("θ₀: ", dynamic_coefficients[1])
println("θ₁: ", dynamic_coefficients[2])
println("θ₂: ", dynamic_coefficients[3])

# Question 4: Unit tests
@testset "Bus Engine Replacement Model Tests" begin
    # Test static model estimation
    @test length(static_coefficients) == 3
    @test all(isfinite.(static_coefficients))

    # Test future value computation
    test_FV = compute_future_values(SVector{3,Float64}(static_coefficients), 0.9, 20, zbin, xbin, xval, xtran, B)
    @test size(test_FV) == (zbin * xbin, 2, 21)
    @test all(isfinite.(test_FV))

    # Test log-likelihood function
    test_ll = log_likelihood(SVector{3,Float64}(static_coefficients), 0.9, 20, zbin, xbin, xval, xtran, Y, Odo, Xst, Zst, B)
    @test isfinite(test_ll)

    # Test dynamic model estimation
    @test length(dynamic_coefficients) == 3
    @test all(isfinite.(dynamic_coefficients))
end

# Precompilation
@setup_workload begin
    @compile_workload begin
        small_Y = Y[1:100, 1:5]
        small_Odo = Odo[1:100, 1:5]
        small_Xst = Xst[1:100, 1:5]
        small_Zst = Zst[1:100]
        small_B = B[1:100]
        estimate_dynamic_model(small_Y, small_Odo, small_Xst, small_Zst, small_B, 0.9, 5)
    end
end