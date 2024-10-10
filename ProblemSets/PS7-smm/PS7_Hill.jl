#Problem Set 7
#Madison Hill

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

# Question 1 
using Optim

function main()
    # Define the function to be maximized
    f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2

    # Define the negative of f(x) for minimization
    negf(x) = -f(x)

    # Set a random starting value
    startval = rand(1)

    # Perform the optimization
    result = optimize(negf, startval, LBFGS())

    # Print the results
    println("Maximizer (argmax): ", Optim.minimizer(result)[1])
    println("Maximum value: ", -Optim.minimum(result))
end

# Run the main function
main()

# Question 2 
# 2a
using DataFrames, CSV, HTTP, GLM, Optim, LinearAlgebra, Random, Statistics

function main()
    # Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Clean the data as specified in Problem Set 2
    df = dropmissing(df, :occupation)
    df[df.occupation .∈ Ref(8:13), :occupation] .= 7
    
    # Prepare the data
    X = Matrix(select(df, :age, :race, :collgrad))
    X = hcat(ones(size(X, 1)), X)  # Add a column of ones for the intercept
    y = df.occupation

    # Define the multinomial logit log-likelihood function
    function mlogit_ll(β, X, y)
        n, k = size(X)
        J = length(unique(y))
        β_matrix = reshape(β, k, J-1)
        
        # Calculate probabilities
        exp_Xβ = exp.(X * β_matrix)
        P = hcat(exp_Xβ, ones(n)) ./ (1 .+ sum(exp_Xβ, dims=2))
        
        # Calculate log-likelihood
        ll = 0.0
        for i in 1:n
            ll += log(P[i, y[i]])
        end
        
        return -ll  # Return negative log-likelihood for minimization
    end

    # Perform MLE estimation
    k = size(X, 2)
    J = length(unique(y))
    initial_β = zeros(k * (J-1))
    
    result = optimize(β -> mlogit_ll(β, X, y), initial_β, LBFGS(), 
                      Optim.Options(show_trace = true, iterations = 1000))

    # Extract and reshape the results
    β_mle = Optim.minimizer(result)
    β_matrix = reshape(β_mle, k, J-1)

    # Print the results
    println("MLE Estimates:")
    println(β_matrix)
end

main()

#2b
using DataFrames, CSV, HTTP, GLM, Optim, LinearAlgebra, Random, Statistics

function main()
    # Load and prepare data (same as before)
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = dropmissing(df, :occupation)
    df[df.occupation .∈ Ref(8:13), :occupation] .= 7
    
    X = Matrix(select(df, :age, :race, :collgrad))
    X = hcat(ones(size(X, 1)), X)  # Add intercept
    y = df.occupation

    # MLE function (same as before)
    function mlogit_ll(β, X, y)
        n, k = size(X)
        J = length(unique(y))
        β_matrix = reshape(β, k, J-1)
        
        exp_Xβ = exp.(X * β_matrix)
        P = hcat(exp_Xβ, ones(n)) ./ (1 .+ sum(exp_Xβ, dims=2))
        
        ll = 0.0
        for i in 1:n
            ll += log(P[i, y[i]])
        end
        
        return -ll
    end

    # Perform MLE estimation to get starting values
    k = size(X, 2)
    J = length(unique(y))
    initial_β = zeros(k * (J-1))
    
    mle_result = optimize(β -> mlogit_ll(β, X, y), initial_β, LBFGS(), 
                          Optim.Options(show_trace = true, iterations = 1000))
    β_mle = Optim.minimizer(mle_result)

    # GMM estimation
    function gmm_objective(β, X, y)
        n, k = size(X)
        J = length(unique(y))
        β_matrix = reshape(β, k, J-1)
        
        exp_Xβ = exp.(X * β_matrix)
        P = hcat(exp_Xβ, ones(n)) ./ (1 .+ sum(exp_Xβ, dims=2))
        
        g = zeros(n, J)
        for j in 1:J
            g[:, j] = (y .== j) .- P[:, j]
        end
        
        return g
    end

    function gmm_criterion(β, X, y, W)
        g = gmm_objective(β, X, y)
        return (mean(g, dims=1) * W * mean(g, dims=1)')[1]
    end

    # Identity matrix as initial weighting matrix
    W = Matrix(I, J, J)

    # GMM estimation using MLE estimates as starting values
    gmm_result = optimize(β -> gmm_criterion(β, X, y, W), β_mle, LBFGS(), 
                          Optim.Options(show_trace = true, iterations = 1000))

    # Extract and reshape the results
    β_gmm = Optim.minimizer(gmm_result)
    β_matrix_gmm = reshape(β_gmm, k, J-1)

    # Print the results
    println("MLE Estimates:")
    println(reshape(β_mle, k, J-1))
    println("\nGMM Estimates:")
    println(β_matrix_gmm)
end

main()

#2c
using DataFrames, CSV, HTTP, GLM, Optim, LinearAlgebra, Random, Statistics

function main()
    # Load and prepare data (same as before)
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = dropmissing(df, :occupation)
    df[df.occupation .∈ Ref(8:13), :occupation] .= 7
    
    X = Matrix(select(df, :age, :race, :collgrad))
    X = hcat(ones(size(X, 1)), X)  # Add intercept
    y = df.occupation

    # GMM functions (same as before)
    function gmm_objective(β, X, y)
        n, k = size(X)
        J = length(unique(y))
        β_matrix = reshape(β, k, J-1)
        
        exp_Xβ = exp.(X * β_matrix)
        P = hcat(exp_Xβ, ones(n)) ./ (1 .+ sum(exp_Xβ, dims=2))
        
        g = zeros(n, J)
        for j in 1:J
            g[:, j] = (y .== j) .- P[:, j]
        end
        
        return g
    end

    function gmm_criterion(β, X, y, W)
        g = gmm_objective(β, X, y)
        return (mean(g, dims=1) * W * mean(g, dims=1)')[1]
    end

    # Set up dimensions
    k = size(X, 2)
    J = length(unique(y))

    # Identity matrix as weighting matrix
    W = Matrix(I, J, J)

    # Generate random starting values
    β_random = randn(k * (J-1))

    # GMM estimation using random starting values
    gmm_result_random = optimize(β -> gmm_criterion(β, X, y, W), β_random, LBFGS(), 
                                 Optim.Options(show_trace = true, iterations = 1000))

    # Extract and reshape the results
    β_gmm_random = Optim.minimizer(gmm_result_random)
    β_matrix_gmm_random = reshape(β_gmm_random, k, J-1)

    # Print the results
    println("GMM Estimates with Random Starting Values:")
    println(β_matrix_gmm_random)

    # Compare objective function values
    println("\nObjective Function Value:")
    println(Optim.minimum(gmm_result_random))
end

main()

#########################################################################
# compare estimates from b and c. Is the objective function globally conccave?
# The estimates from part b and part c are different, which suggests that the objective function is not globally concave.
#########################################################################

#question 3
using Random, Distributions, LinearAlgebra, Optim, Printf

# Part (a): Choose N, J, K, and β
N = 1000  # Sample size
J = 4     # Number of choices
K = 3     # Number of covariates (excluding intercept)

# Set parameter values
β = [0.5 1.0 -0.5 0.0;   # Intercept
     1.0 -0.5 0.2 0.0;   # First covariate
     -0.5 0.8 -0.3 0.0;  # Second covariate
     0.2 -0.3 0.6 0.0]   # Third covariate

# Part (b): Write a function that outputs data X and Y
function generate_multinomial_logit_data(N::Int, J::Int, K::Int, β::Matrix{Float64})
    # Part (c): Generate X using a random number generator
    X = randn(N, K)
    X = hcat(ones(N), X)  # Add intercept
    
    # Part (d): Draw a N × J matrix of ε's from a T1EV distribution
    ε = rand(Gumbel(0, 1), N, J)
    
    # Calculate utility
    V = X * β
    U = V + ε
    
    # Part (e): Choose Y_i = argmax_j (X_iβ_j + ε_ij)
    Y = [argmax(U[i,:]) for i in 1:N]
    
    return X, Y
end

# Generate the data
X, Y = generate_multinomial_logit_data(N, J, K, β)

# Define log-likelihood function for estimation
function mlogit_loglikelihood(β::Vector{Float64}, X::Matrix{Float64}, Y::Vector{Int}, J::Int)
    N, K = size(X)
    β_matrix = reshape(β, K, J-1)
    V = X * β_matrix
    V = hcat(V, zeros(N))  # Add zeros for base category
    P = exp.(V) ./ sum(exp.(V), dims=2)
    
    ll = 0.0
    for i in 1:N
        ll += log(P[i, Y[i]])
    end
    return -ll  # Return negative log-likelihood for minimization
end

# Estimate parameters
K_total = size(X, 2)
β_init = zeros(K_total * (J-1))  # Initial values

result = optimize(β -> mlogit_loglikelihood(β, X, Y, J), β_init, LBFGS(),
                  Optim.Options(show_trace=true, iterations=1000))

# Extract estimated parameters
β_est = reshape(Optim.minimizer(result), K_total, J-1)

# Compare true and estimated parameters
println("True vs. Estimated Parameters:")
for k in 1:K_total
    for j in 1:(J-1)
        @printf("β[%d,%d]: True = %.4f, Estimated = %.4f, Difference = %.4f\n", 
                k, j, β[k,j], β_est[k,j], β_est[k,j] - β[k,j])
    end
end

# Print choice frequencies
println("\nChoice Frequencies:")
for j in 1:J
    @printf("Choice %d: %.2f%%\n", j, 100 * count(Y .== j) / N)
end

# skip question 4

# question 5
using DataFrames, CSV, HTTP, Random, Distributions, LinearAlgebra, Optim, Statistics, Printf

function q5_manual_smm_estimation()
    # Load and prepare data (same as in Question 2)
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = dropmissing(df, :occupation)
    df[df.occupation .∈ Ref(8:13), :occupation] .= 7
    X = Matrix(select(df, :age, :race, :collgrad))
    X = hcat(ones(size(X, 1)), X)
    y = df.occupation

    function generate_multinomial_logit_data(X, β)
        N, K = size(X)
        J = size(β, 2)
        ε = rand(Gumbel(0, 1), N, J)
        V = X * β
        U = V + ε
        Y = [argmax(U[i,:]) for i in 1:N]
        return Y
    end

    function moment_function(data, sim_data)
        N = length(data)
        J = maximum(data)
        g = zeros(N, J)
        for j in 1:J
            g[:, j] = (data .== j) - (sim_data .== j)
        end
        return vec(mean(g, dims=1))
    end

    function manual_smm_objective(β, X, y, S)
        N, K = size(X)
        J = maximum(y)
        β_matrix = reshape(β, K, J)
        moment_sum = zeros(J)
        for s in 1:S
            sim_y = generate_multinomial_logit_data(X, β_matrix)
            moment_sum += moment_function(y, sim_y)
        end
        moments = moment_sum / S
        return moments' * moments
    end

    S = 10
    J = maximum(y)
    K = size(X, 2)
    β_init = zeros(K * J)

    result = optimize(β -> manual_smm_objective(β, X, y, S), β_init, LBFGS(),
                      Optim.Options(show_trace=false))

    β_smm = reshape(Optim.minimizer(result), K, J)

    println("Question 5 Results (Manual SMM Estimates):")
    println(β_smm)
    println()
end


#question 6
#wrap all the code in a function
using DataFrames, CSV, HTTP, Random, Distributions, LinearAlgebra, Optim, Statistics, Printf

function main()
    # Question 1: Basic optimization
    function q1_optimization()
        f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
        negf(x) = -f(x)
        startval = rand(1)
        result = optimize(negf, startval, LBFGS())
        println("Question 1 Results:")
        println("Maximizer: ", Optim.minimizer(result)[1])
        println("Maximum value: ", -Optim.minimum(result))
        println()
    end

    # Question 2: Multinomial Logit Estimation (GMM)
    function q2_multinomial_logit_gmm()
        # Load and prepare data
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        df = dropmissing(df, :occupation)
        df[df.occupation .∈ Ref(8:13), :occupation] .= 7
        X = Matrix(select(df, :age, :race, :collgrad))
        X = hcat(ones(size(X, 1)), X)
        y = df.occupation

        function gmm_objective(β, X, y)
            N, K = size(X)
            J = maximum(y)
            β_matrix = reshape(β, K, J-1)
            V = X * β_matrix
            V = hcat(V, zeros(N))
            P = exp.(V) ./ sum(exp.(V), dims=2)
            g = zeros(N, J)
            for j in 1:J
                g[:, j] = (y .== j) .- P[:, j]
            end
            return mean(g, dims=1)
        end

        function gmm_criterion(β, X, y, W)
            g = gmm_objective(β, X, y)
            return (g * W * g')[1]
        end

        J = maximum(y)
        K = size(X, 2)
        W = Matrix(I, J, J)
        β_init = zeros(K * (J-1))

        result = optimize(β -> gmm_criterion(β, X, y, W), β_init, LBFGS(),
                          Optim.Options(show_trace=false))

        β_gmm = reshape(Optim.minimizer(result), K, J-1)
        println("Question 2 Results (GMM Estimates):")
        println(β_gmm)
        println()
    end

    # Question 3: Simulate and Estimate Multinomial Logit
    function q3_simulate_estimate_mlogit()
        N = 1000
        J = 4
        K = 3
        β_true = [0.5 1.0 -0.5 0.0;
                  1.0 -0.5 0.2 0.0;
                  -0.5 0.8 -0.3 0.0;
                  0.2 -0.3 0.6 0.0]

        function generate_multinomial_logit_data(N, J, K, β)
            X = randn(N, K)
            X = hcat(ones(N), X)
            ε = rand(Gumbel(0, 1), N, J)
            V = X * β
            U = V + ε
            Y = [argmax(U[i,:]) for i in 1:N]
            return X, Y
        end

        X, Y = generate_multinomial_logit_data(N, J, K, β_true)

        function mlogit_loglikelihood(β, X, Y, J)
            N, K = size(X)
            β_matrix = reshape(β, K, J-1)
            V = X * β_matrix
            V = hcat(V, zeros(N))
            P = exp.(V) ./ sum(exp.(V), dims=2)
            ll = sum(log.(P[i, Y[i]]) for i in 1:N)
            return -ll
        end

        β_init = zeros(size(X, 2) * (J-1))
        result = optimize(β -> mlogit_loglikelihood(β, X, Y, J), β_init, LBFGS())
        β_est = reshape(Optim.minimizer(result), size(X, 2), J-1)

        println("Question 3 Results:")
        println("True vs. Estimated Parameters:")
        for k in 1:size(X, 2)
            for j in 1:(J-1)
                @printf("β[%d,%d]: True = %.4f, Estimated = %.4f, Difference = %.4f\n",
                        k, j, β_true[k,j], β_est[k,j], β_est[k,j] - β_true[k,j])
            end
        end
        println()
    end

    # Question 5: SMM Estimation of Multinomial Logit
    function q5_smm_estimation()
        # Load and prepare data (same as in Question 2)
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        df = dropmissing(df, :occupation)
        df[df.occupation .∈ Ref(8:13), :occupation] .= 7
        X = Matrix(select(df, :age, :race, :collgrad))
        X = hcat(ones(size(X, 1)), X)
        y = df.occupation

        function generate_multinomial_logit_data(X, β)
            N, K = size(X)
            J = size(β, 2)
            ε = rand(Gumbel(0, 1), N, J)
            V = X * β
            U = V + ε
            Y = [argmax(U[i,:]) for i in 1:N]
            return Y
        end

        function moment_function(data, sim_data)
            N = length(data)
            J = maximum(data)
            g = zeros(N, J)
            for j in 1:J
                g[:, j] = (data .== j) - (sim_data .== j)
            end
            return vec(mean(g, dims=1))
        end

        function smm_objective(β, X, y, S)
            N, K = size(X)
            J = maximum(y)
            β_matrix = reshape(β, K, J)
            moment_sum = zeros(J)
            for s in 1:S
                sim_y = generate_multinomial_logit_data(X, β_matrix)
                moment_sum += moment_function(y, sim_y)
            end
            moments = moment_sum / S
            return moments' * moments
        end

        S = 10
        J = maximum(y)
        K = size(X, 2)
        β_init = zeros(K * J)

        result = optimize(β -> smm_objective(β, X, y, S), β_init, LBFGS(),
                          Optim.Options(show_trace=false))

        β_smm = reshape(Optim.minimizer(result), K, J)

        println("Question 5 Results (SMM Estimates):")
        println(β_smm)
        println()
    end

    # Run all questions
    q1_optimization()
    q2_multinomial_logit_gmm()
    q3_simulate_estimate_mlogit()
    q5_smm_estimation()
end

# Run the main function
main()

# question 7
using Test, Random, Distributions, LinearAlgebra, Optim, Statistics, DataFrames, CSV, HTTP

# Define the necessary functions here
function generate_multinomial_logit_data(X::Matrix{Float64}, β::Matrix{Float64})
    N, K = size(X)
    J = size(β, 2)
    ε = rand(Gumbel(0, 1), N, J)
    V = X * β
    U = V + ε
    Y = [argmax(U[i,:]) for i in 1:N]
    return Y
end

function moment_function(data, sim_data)
    N = length(data)
    J = maximum(data)
    g = zeros(N, J)
    for j in 1:J
        g[:, j] = (data .== j) - (sim_data .== j)
    end
    return vec(mean(g, dims=1))
end

@testset "Problem Set 7 Tests" begin
    @testset "Question 1: Optimization" begin
        f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
        negf(x) = -f(x)
        result = optimize(negf, [0.0], LBFGS())
        @test isapprox(Optim.minimizer(result)[1], -7.37824, atol=1e-5)
        @test isapprox(-Optim.minimum(result), 964.313384, atol=1e-5)
    end

    @testset "Question 2: Multinomial Logit GMM" begin
        # Create a simple test dataset
        X_test = [ones(100) randn(100, 2)]
        β_true = [1.0 0.5; -0.5 1.0; 0.2 -0.3]
        y_test = [argmax(X_test[i,:] * β_true + rand(Gumbel(), 3)) for i in 1:100]

        function gmm_objective_test(β, X, y)
            N, K = size(X)
            J = maximum(y)
            β_matrix = reshape(β, K, J-1)
            V = X * β_matrix
            V = hcat(V, zeros(N))
            P = exp.(V) ./ sum(exp.(V), dims=2)
            g = zeros(N, J)
            for j in 1:J
                g[:, j] = (y .== j) .- P[:, j]
            end
            return mean(g, dims=1)
        end

        g = gmm_objective_test(vec(β_true[:,1:2]), X_test, y_test)
        @test size(g) == (1, 3)
        @test all(abs.(g) .< 0.1)  # Moments should be close to zero
    end

    @testset "Question 3: Simulate and Estimate Multinomial Logit" begin
        N, J, K = 1000, 4, 3
        β_true = [0.5 1.0 -0.5 0.0;
                  1.0 -0.5 0.2 0.0;
                  -0.5 0.8 -0.3 0.0;
                  0.2 -0.3 0.6 0.0]

        X = [ones(N) randn(N, K)]
        Y = generate_multinomial_logit_data(X, β_true)
        
        @test size(X) == (N, K+1)  # K+1 because of intercept
        @test length(Y) == N
        @test all(1 .<= Y .<= J)

        function mlogit_loglikelihood_test(β, X, Y, J)
            N, K = size(X)
            β_matrix = reshape(β, K, J-1)
            V = X * β_matrix
            V = hcat(V, zeros(N))
            P = exp.(V) ./ sum(exp.(V), dims=2)
            ll = sum(log.(P[i, Y[i]]) for i in 1:N)
            return -ll
        end

        β_init = zeros(size(X, 2) * (J-1))
        result = optimize(β -> mlogit_loglikelihood_test(β, X, Y, J), β_init, LBFGS())
        β_est = reshape(Optim.minimizer(result), size(X, 2), J-1)

        # Check if estimated parameters are close to true parameters
        @test isapprox(β_est, β_true[:, 1:3], atol=0.5)
    end

    @testset "Question 5: Manual SMM Estimation" begin
        # Create a simple test dataset
        N, J, K = 1000, 4, 3
        X_test = [ones(N) randn(N, K)]
        β_true = [0.5 1.0 -0.5 0.0;
                  1.0 -0.5 0.2 0.0;
                  -0.5 0.8 -0.3 0.0;
                  0.2 -0.3 0.6 0.0]
        y_test = generate_multinomial_logit_data(X_test, β_true)

        function manual_smm_objective_test(β, X, y, S)
            N, K = size(X)
            J = maximum(y)
            β_matrix = reshape(β, K, J)
            moment_sum = zeros(J)
            for s in 1:S
                sim_y = generate_multinomial_logit_data(X, β_matrix)
                moment_sum += moment_function(y, sim_y)
            end
            moments = moment_sum / S
            return moments' * moments
        end

        S = 5
        β_init = zeros(size(X_test, 2) * J)
        result = optimize(β -> manual_smm_objective_test(β, X_test, y_test, S), β_init, LBFGS(),
                          Optim.Options(iterations=100))
        β_smm = reshape(Optim.minimizer(result), size(X_test, 2), J)

        # Check if manual SMM estimates are reasonably close to true parameters
        @test isapprox(β_smm, β_true, atol=0.5)
    end
end
