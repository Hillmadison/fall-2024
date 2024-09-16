# Problem Set 3
# Madison Hill

using Optim, LinearAlgebra, Random, Statistics, DataFrames, CSV, HTTP

# Question 1

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

# Define the likelihood function
function mnl_loglikelihood(β, X, Z, y)
    n, k = size(X)
    J = size(Z, 2)
    
    ll = 0.0
    for i in 1:n
        denom = 1.0
        for j in 1:(J-1)
            denom += exp(X[i,:]' * β[((j-1)*k+1):(j*k)] + β[end] * (Z[i,j] - Z[i,J]))
        end
        
        if y[i] == J
            ll += log(1 / denom)
        else
            ll += X[i,:]' * β[((y[i]-1)*k+1):(y[i]*k)] + β[end] * (Z[i,y[i]] - Z[i,J]) - log(denom)
        end
    end
    
    return -ll  # Return negative log-likelihood for minimization
end

# Set up initial parameters and optimize
k = size(X, 2)
J = size(Z, 2)
initial_β = vcat(repeat([0.0], (J-1)*k), [0.0])  # Initialize all parameters to 0

result = optimize(β -> mnl_loglikelihood(β, X, Z, y), initial_β, BFGS())

# Extract and print results
estimated_β = Optim.minimizer(result)
println("Estimated parameters:")
println(estimated_β)

# Question 2
# Interpreting the estimated coefficient γ̂: 
# y hat is the effect of alternative-specific covariates on our choice probabilities.
# A positive value of y indiaces that higher wages increase the chance of choosing a job

# Question 3
function nested_logit_loglikelihood(θ, X, Z, y, nest_structure)
    n, k = size(X)
    J = size(Z, 2)
    
    # Unpack parameters
    β_WC = θ[1:k]
    β_BC = θ[k+1:2k]
    λ_WC = θ[2k+1]
    λ_BC = θ[2k+2]
    γ = θ[end]
    
    ll = 0.0
    for i in 1:n
        WC_sum = sum(exp((X[i,:]' * β_WC + γ * (Z[i,j] - Z[i,J])) / λ_WC) for j in nest_structure["WC"])
        BC_sum = sum(exp((X[i,:]' * β_BC + γ * (Z[i,j] - Z[i,J])) / λ_BC) for j in nest_structure["BC"])
        
        denom = 1 + WC_sum^λ_WC + BC_sum^λ_BC
        
        if y[i] in nest_structure["WC"]
            ll += (X[i,:]' * β_WC + γ * (Z[i,y[i]] - Z[i,J])) / λ_WC + (λ_WC - 1) * log(WC_sum) - log(denom)
        elseif y[i] in nest_structure["BC"]
            ll += (X[i,:]' * β_BC + γ * (Z[i,y[i]] - Z[i,J])) / λ_BC + (λ_BC - 1) * log(BC_sum) - log(denom)
        else  # Other category
            ll += -log(denom)
        end
    end
    
    return -ll  # Return negative log-likelihood for minimization
end

# Define nest structure
nest_structure = Dict(
    "WC" => [1, 2, 3],
    "BC" => [4, 5, 6, 7],
    "Other" => [8]
)

# Set up initial parameters and optimize
k = size(X, 2)
initial_θ = vcat(repeat([0.0], 2k), [1.0, 1.0], [0.0])  # Initialize β's to 0, λ's to 1, and γ to 0

result_nested = optimize(θ -> nested_logit_loglikelihood(θ, X, Z, y, nest_structure), initial_θ, BFGS())

# Extract and print results
estimated_θ = Optim.minimizer(result_nested)
println("Estimated parameters for nested logit:")
println(estimated_θ)

# Question 4

function estimate_models(X, Z, y, nest_structure)
    # Multinomial Logit
    k = size(X, 2)
    J = size(Z, 2)
    initial_β = vcat(repeat([0.0], (J-1)*k), [0.0])
    result_mnl = optimize(β -> mnl_loglikelihood(β, X, Z, y), initial_β, BFGS())
    estimated_β = Optim.minimizer(result_mnl)
    
    println("Estimated parameters for multinomial logit:")
    println(estimated_β)
    
    # Nested Logit
    initial_θ = vcat(repeat([0.0], 2k), [1.0, 1.0], [0.0])
    result_nested = optimize(θ -> nested_logit_loglikelihood(θ, X, Z, y, nest_structure), initial_θ, BFGS())
    estimated_θ = Optim.minimizer(result_nested)
    
    println("Estimated parameters for nested logit:")
    println(estimated_θ)
    
    return estimated_β, estimated_θ
end

# Call the function
nest_structure = Dict("WC" => [1, 2, 3], "BC" => [4, 5, 6, 7], "Other" => [8])
mnl_results, nested_results = estimate_models(X, Z, y, nest_structure)

# Question 5

using Test

@testset "Econometrics Models Tests" begin
    # Test data
    X_test = rand(100, 3)
    Z_test = rand(100, 8)
    y_test = rand(1:8, 100)
    nest_structure_test = Dict("WC" => [1, 2, 3], "BC" => [4, 5, 6, 7], "Other" => [8])

    # Test multinomial logit likelihood
    @test typeof(mnl_loglikelihood(zeros(3*7+1), X_test, Z_test, y_test)) == Float64
    
    # Test nested logit likelihood
    @test typeof(nested_logit_loglikelihood(zeros(2*3+3), X_test, Z_test, y_test, nest_structure_test)) == Float64
    
    # Test full estimation function
    mnl_results, nested_results = estimate_models(X_test, Z_test, y_test, nest_structure_test)
    @test length(mnl_results) == 3*7+1
    @test length(nested_results) == 2*3+3
    
    # Test that probabilities sum to 1 for a single observation
    function calc_probs_mnl(β, x, z)
        J = size(z, 1)
        probs = zeros(J)
        denom = 1.0
        for j in 1:(J-1)
            probs[j] = exp(x' * β[((j-1)*3+1):(j*3)] + β[end] * (z[j] - z[J]))
            denom += probs[j]
        end
        probs[J] = 1.0
        probs ./= denom
        return probs
    end
    
    probs = calc_probs_mnl(mnl_results, X_test[1,:], Z_test[1,:])
    @test isapprox(sum(probs), 1.0, atol=1e-6)
end