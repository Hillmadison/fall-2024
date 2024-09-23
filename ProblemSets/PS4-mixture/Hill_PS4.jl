# Problem Set 4
# Madison Hill

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, 
      Distributions, ForwardDiff, Test

include("lgwt.jl")  # For Gauss-Legendre quadrature

# Question 1

function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code
    return df, X, Z, y
end

function mlogit_with_Z(theta, X, Z, y)
    alpha = theta[1:end-1]
    gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = size(X,1)
    
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    T = promote_type(eltype(X),eltype(theta))
    num   = zeros(T,N,J)
    dem   = zeros(T,N)
    
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)
    
    loglike = -sum(bigY.*log.(P))
    
    return loglike
end

function optimize_mlogit(X, Z, y)
    startvals = [2*rand(7*size(X,2)).-1; .1]
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    return theta_hat_mle_ad, theta_hat_mle_ad_se
end

# Question 2
# Gamma hat makes more sense in this problem set then in the previous one because in the previous one gamma hat was negative and now it is positive. 

# Question 3

using Distributions, LinearAlgebra, Random, Statistics

include("lgwt.jl")

# Part (a): Practice using quadrature
d = Normal(0, 1)

nodes, weights = lgwt(7, -4, 4)

integral_density = sum(weights .* pdf.(d, nodes))
println("Integral of density: ", integral_density)

expectation = sum(weights .* nodes .* pdf.(d, nodes))
println("Expectation: ", expectation)

# Part (b): More practice with quadrature
d_b = Normal(0, 2)

nodes_7, weights_7 = lgwt(7, -5*sqrt(2), 5*sqrt(2))
integral_7 = sum(weights_7 .* nodes_7.^2 .* pdf.(d_b, nodes_7))
println("Integral with 7 points: ", integral_7)

nodes_10, weights_10 = lgwt(10, -5*sqrt(2), 5*sqrt(2))
integral_10 = sum(weights_10 .* nodes_10.^2 .* pdf.(d_b, nodes_10))
println("Integral with 10 points: ", integral_10)

println("True variance: ", var(d_b))

# Part (c): Monte Carlo integration
function monte_carlo_integral(f, a, b, D)
    x = rand(Uniform(a, b), D)
    return (b - a) * mean(f.(x))
end

Random.seed!(123)

# N(0, 2) distribution
d_c = Normal(0, 2)

# Integrate x^2 * f(x)
mc_integral_1m = monte_carlo_integral(x -> x^2 * pdf(d_c, x), -5*sqrt(2), 5*sqrt(2), 1_000_000)
println("Monte Carlo integral of x^2 * f(x) with 1,000,000 draws: ", mc_integral_1m)

# Integrate x * f(x)
mc_integral_x = monte_carlo_integral(x -> x * pdf(d_c, x), -5*sqrt(2), 5*sqrt(2), 1_000_000)
println("Monte Carlo integral of x * f(x) with 1,000,000 draws: ", mc_integral_x)

# Integrate f(x)
mc_integral_f = monte_carlo_integral(x -> pdf(d_c, x), -5*sqrt(2), 5*sqrt(2), 1_000_000)
println("Monte Carlo integral of f(x) with 1,000,000 draws: ", mc_integral_f)

# Compare with 1,000 draws
mc_integral_1k = monte_carlo_integral(x -> x^2 * pdf(d_c, x), -5*sqrt(2), 5*sqrt(2), 1_000)
println("Monte Carlo integral of x^2 * f(x) with 1,000 draws: ", mc_integral_1k)

# Question 4

include("lgwt.jl")  

function mixed_logit_prob(X_i, Z_i, beta, gamma, J)
    K = length(X_i)
    U = zeros(J)
    for j in 1:J-1
        U[j] = dot(X_i, beta[(j-1)*K+1:j*K]) + gamma * (Z_i[j] - Z_i[J])
    end
    # U[J] is left as 0 (reference alternative)
    
    exp_U = exp.(U)
    probs = exp_U ./ sum(exp_U)
    
    return probs
end

function mixed_logit_likelihood_quadrature(theta, X, Z, y, num_quad_points)
    beta = theta[1:end-2]
    mu_gamma = theta[end-1]
    sigma_gamma = exp(theta[end])  # Use exp to ensure sigma_gamma is positive
    
    K = size(X,2)
    J = length(unique(y))
    N = size(X,1)
    
    # Get quadrature nodes and weights
    nodes, weights = lgwt(num_quad_points, -4, 4)  # Use standard normal range
    
    loglike = 0.0
    
    for i in 1:N
        prob_i = 0.0
        for (node, weight) in zip(nodes, weights)
            gamma_q = mu_gamma + sigma_gamma * node
            probs = mixed_logit_prob(X[i,:], Z[i,:], beta, gamma_q, J)
            prob_i += weight * probs[y[i]]
        end
        loglike -= log(prob_i)
    end
    
    return loglike
end

function estimate_mixed_logit_quadrature(X, Z, y, num_quad_points)
    K = size(X,2)
    J = length(unique(y))
    
    initial_theta = [rand(K*(J-1)); 0.0; log(0.1)]  # Initialize sigma_gamma as log(0.1)
    
    result = optimize(theta -> mixed_logit_likelihood_quadrature(theta, X, Z, y, num_quad_points),
                      initial_theta, LBFGS(), Optim.Options(iterations=1000))
    
    final_theta = result.minimizer
    final_theta[end] = exp(final_theta[end])  # Transform sigma_gamma back to original scale
    
    return final_theta
end

# D note the simularity between quadrature and Monte Carlo
# Both methods aim to approximate the value of integrals by summing over a set of points (nodes) weighted by some weights.
# The main difference is that quadrature uses a fixed set of nodes and weights, while Monte Carlo uses random draws from a distribution.
# both also are flexible and can be applied to a large range of integrals 

# Question 5

function mixed_logit_prob(X_i, Z_i, beta, gamma, J)
    K = length(X_i)
    U = zeros(J)
    for j in 1:J-1
        U[j] = dot(X_i, beta[(j-1)*K+1:j*K]) + gamma * (Z_i[j] - Z_i[J])
    end
    # U[J] is left as 0 (reference alternative)
    
    exp_U = exp.(U)
    probs = exp_U ./ sum(exp_U)
    
    return probs
end

function mixed_logit_likelihood_monte_carlo(theta, X, Z, y, num_draws)
    beta = theta[1:end-2]
    mu_gamma = theta[end-1]
    sigma_gamma = exp(theta[end])  # Use exp to ensure sigma_gamma is positive
    
    K = size(X,2)
    J = length(unique(y))
    N = size(X,1)
    
    loglike = 0.0
    
    for i in 1:N
        prob_i = 0.0
        for _ in 1:num_draws
            gamma_mc = rand(Normal(mu_gamma, sigma_gamma))
            probs = mixed_logit_prob(X[i,:], Z[i,:], beta, gamma_mc, J)
            prob_i += probs[y[i]]
        end
        loglike -= log(prob_i / num_draws)
    end
    
    return loglike
end

function estimate_mixed_logit_monte_carlo(X, Z, y, num_draws)
    K = size(X,2)
    J = length(unique(y))
    
    initial_theta = [rand(K*(J-1)); 0.0; log(0.1)]  # Initialize sigma_gamma as log(0.1)
    
    result = optimize(theta -> mixed_logit_likelihood_monte_carlo(theta, X, Z, y, num_draws),
                      initial_theta, LBFGS(), Optim.Options(iterations=1000))
    
    final_theta = result.minimizer
    final_theta[end] = exp(final_theta[end])  # Transform sigma_gamma back to original scale
    
    return final_theta
end

# Question 6

function run_all_estimations()
    # Load data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df, X, Z, y = load_data(url)

    # Question 1: Multinomial Logit
    theta_mlogit, se_mlogit = optimize_mlogit(X, Z, y)
    println("Multinomial Logit Results:")
    println("Estimates: ", theta_mlogit)
    println("Standard Errors: ", se_mlogit)

    # Question 3: Mixed Logit
    # Use the Monte Carlo version from Question 5
    theta_mixed = estimate_mixed_logit_monte_carlo(X, Z, y, 100)  # Using 100 draws
    println("\nMixed Logit Results (Monte Carlo):")
    println(theta_mixed)

    println("\nFunctions for Questions 4 and 5 are used in the Mixed Logit estimation above.")
end


run_all_estimations()

# Question 7

@testset "Problem Set 4 Tests" begin
    # Test data loading
    @test begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
        df, X, Z, y = load_data(url)
        !isempty(df) && size(X, 2) == 3 && size(Z, 2) == 8 && length(y) == size(X, 1)
    end

    # Test multinomial logit function
    @test begin
        X_test = rand(100, 3)
        Z_test = rand(100, 8)
        y_test = rand(1:8, 100)
        theta_test = rand(22)  # 21 coefficients + 1 gamma
        result = mlogit_with_Z(theta_test, X_test, Z_test, y_test)
        typeof(result) <: Real && !isnan(result)
    end

    # Test mixed logit probability function
    @test begin
        X_i = rand(3)
        Z_i = rand(8)
        beta = rand(21)
        gamma = rand()
        J = 8
        probs = mixed_logit_prob(X_i, Z_i, beta, gamma, J)
        length(probs) == J && all(0 .<= probs .<= 1) && isapprox(sum(probs), 1, atol=1e-6)
    end

    # Test mixed logit likelihood function
    @test begin
        X_test = rand(100, 3)
        Z_test = rand(100, 8)
        y_test = rand(1:8, 100)
        theta_test = rand(24)  # 21 coefficients + mu_gamma + sigma_gamma
        num_draws = 10
        result = mixed_logit_likelihood(theta_test, X_test, Z_test, y_test, num_draws)
        typeof(result) <: Real && !isnan(result)
    end

    # Test Gauss-Legendre quadrature function
    @test begin
        nodes, weights = lgwt(5, -1, 1)
        length(nodes) == 5 && length(weights) == 5 && 
        all(-1 .<= nodes .<= 1) && isapprox(sum(weights), 2, atol=1e-6)
    end

    # Test mixed logit with quadrature
    @test begin
        X_test = rand(100, 3)
        Z_test = rand(100, 8)
        y_test = rand(1:8, 100)
        theta_test = rand(24)  # 21 coefficients + mu_gamma + sigma_gamma
        num_quad_points = 7
        result = mixed_logit_likelihood_quadrature(theta_test, X_test, Z_test, y_test, num_quad_points)
        typeof(result) <: Real && !isnan(result)
    end

    # Test mixed logit with Monte Carlo
    @test begin
        X_test = rand(100, 3)
        Z_test = rand(100, 8)
        y_test = rand(1:8, 100)
        theta_test = rand(24)  # 21 coefficients + mu_gamma + sigma_gamma
        num_draws = 100
        result = mixed_logit_likelihood_monte_carlo(theta_test, X_test, Z_test, y_test, num_draws)
        typeof(result) <: Real && !isnan(result)
    end

    # Test optimization functions (these might take a while to run)
    @testset "Optimization Tests" begin
        X_small = rand(50, 3)
        Z_small = rand(50, 8)
        y_small = rand(1:8, 50)

        @test begin
            result = optimize_mlogit(X_small, Z_small, y_small)
            length(result) == 2 && length(result[1]) == 22 && length(result[2]) == 22
        end

        @test begin
            result = estimate_mixed_logit(X_small, Z_small, y_small, 10)
            length(result) == 24
        end

        @test begin
            result = estimate_mixed_logit_quadrature(X_small, Z_small, y_small, 5)
            length(result) == 24
        end

        @test begin
            result = estimate_mixed_logit_monte_carlo(X_small, Z_small, y_small, 50)
            length(result) == 24
        end
    end
end