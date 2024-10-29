# Problem Set 8
# Madison Hill 

using DataFrames, CSV, GLM, Statistics, DataFramesMeta
using HTTP, LinearAlgebra, Random, MultivariateStats, Optim

data = CSV.read("ProblemSets/PS8-factor/nlsy.csv", DataFrame)

# Question 1
model1 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), data)

println("\nRegression Results:")
println(model1)

# Question 2

using DataFrames, CSV, GLM, Statistics, DataFramesMeta, Printf
using HTTP, LinearAlgebra, Random, MultivariateStats, Optim

data = CSV.read("ProblemSets/PS8-factor/nlsy.csv", DataFrame)

asvab_vars = ["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]
asvab_matrix = Matrix(data[:, asvab_vars])

cor_matrix = cor(asvab_matrix)

println("\nCorrelation Matrix of ASVAB Variables:")
println("Variables: AR = Armed Forces Qualification Test (AFQT)")
println("          CS = Coding Speed")
println("          MK = Math Knowledge")
println("          NO = Numerical Operations")
println("          PC = Paragraph Comprehension")
println("          WK = Word Knowledge\n")

println("       AR      CS      MK      NO      PC      WK")
for (i, var) in enumerate(asvab_vars)
    var_short = split(var, "asvab")[2]
    print(rpad(var_short, 8))
    for j in 1:6
        print(lpad(round(cor_matrix[i,j], digits=3), 7))
    end
    println()
end

# Question 3

using DataFrames, CSV, GLM, Statistics, DataFramesMeta, Printf
using HTTP, LinearAlgebra, Random, MultivariateStats, Optim

data = CSV.read("ProblemSets/PS8-factor/nlsy.csv", DataFrame)

model3 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
                    asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), data)

println("\nRegression Results with ASVAB Variables:")
println(model3)

println("\nASVAB Variables Correlation Matrix (for multicollinearity check):")
asvab_vars = ["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]
asvab_matrix = Matrix(data[:, asvab_vars])
cor_matrix = cor(asvab_matrix)

println("\n       AR      CS      MK      NO      PC      WK")
for (i, var) in enumerate(asvab_vars)
    var_short = split(var, "asvab")[2]
    print(rpad(var_short, 8))
    for j in 1:6
        print(lpad(round(cor_matrix[i,j], digits=3), 7))
    end
    println()
end

model1 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), data)
println("\nR² comparison:")
println("Original model R²: ", round(r2(model1), digits=3))
println("Model with ASVAB R²: ", round(r2(model3), digits=3))

# Question 4 AND 5

using DataFrames, CSV, GLM, Statistics, DataFramesMeta, Printf
using HTTP, LinearAlgebra, Random, MultivariateStats, Optim

data = CSV.read("ProblemSets/PS8-factor/nlsy.csv", DataFrame)

asvab_vars = ["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]
asvab_matrix = Matrix(data[:, asvab_vars])

asvab_std = (asvab_matrix .- mean(asvab_matrix, dims=1)) ./ std(asvab_matrix, dims=1)
asvab_std_t = transpose(asvab_std)

M_pca = fit(PCA, asvab_std_t; maxoutdim=1)
pc1 = MultivariateStats.transform(M_pca, asvab_std_t)
data.asvab_pc1 = vec(transpose(pc1))

M_fa = fit(FactorAnalysis, asvab_std_t; maxoutdim=1)
fa1 = MultivariateStats.transform(M_fa, asvab_std_t)
data.asvab_factor = vec(transpose(fa1))

model1 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), data)
model3 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
                    asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), data)
model4 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvab_pc1), data)
model5 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvab_factor), data)

println("\nPCA Results:")
println("Proportion of variance explained by PC1: ", round(principalratio(M_pca), digits=3))
println("\nPCA Loadings:")
for (var, loading) in zip(asvab_vars, projection(M_pca))
    println(rpad(var, 10), ": ", round(loading, digits=3))
end

println("\nFactor Analysis Results:")
println("Factor Loadings:")
for (var, loading) in zip(asvab_vars, loadings(M_fa))
    println(rpad(var, 10), ": ", round(loading, digits=3))
end

println("\nR² Comparison:")
println("Original model R²: ", round(r2(model1), digits=3))
println("Model with all ASVAB variables R²: ", round(r2(model3), digits=3))
println("Model with first principal component R²: ", round(r2(model4), digits=3))
println("Model with factor analysis R²: ", round(r2(model5), digits=3))

println("\nRegression Results with Factor Analysis:")
println(model5)

# Question 6

using DataFrames, CSV, GLM, Statistics, DataFramesMeta
using HTTP, LinearAlgebra, Random, MultivariateStats, Optim
using Distributions, FastGaussQuadrature

data = CSV.read("ProblemSets/PS8-factor/nlsy.csv", DataFrame)
n_sample = 500  # Much smaller sample for faster convergence
Random.seed!(123)  # For reproducibility
sample_indices = Random.shuffle(1:nrow(data))[1:n_sample]
data_sample = data[sample_indices, :]

asvab_vars = ["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]
M = Matrix(data_sample[:, asvab_vars])
M_std = (M .- mean(M, dims=1)) ./ std(M, dims=1)

function create_X_matrices(data)
    X_m = hcat(ones(nrow(data)), 
               Float64.(data.black), 
               Float64.(data.hispanic), 
               Float64.(data.female))
    
    X = hcat(X_m, 
             Float64.(data.schoolt),
             Float64.(data.gradHS),
             Float64.(data.grad4yr))
    
    return X_m, X
end

X_m, X = create_X_matrices(data_sample)

n_quad = 7  # Reduced from previous versions
nodes, weights = gausslegendre(n_quad)
quad_nodes = √2 .* nodes
quad_weights = weights ./ √π

function calculate_measurement_likelihood(X_m_i, M_std_i, α_m, γ, σ_m, ξ)
    n_asvab = length(γ)
    meas_lik = 0.0
    
    for j in 1:n_asvab
        μ_m = dot(X_m_i, α_m[j,:]) + γ[j]*ξ
        meas_lik += logpdf(Normal(μ_m, σ_m[j]), M_std_i[j])
    end
    
    return exp(meas_lik)  # Convert back from log space
end

function log_likelihood(params::Vector{Float64}, M_std::Matrix{Float64}, 
                       X_m::Matrix{Float64}, X::Matrix{Float64}, 
                       y::Vector{Float64}, nodes::Vector{Float64}, 
                       weights::Vector{Float64})
    n_obs = size(M_std, 1)
    n_asvab = size(M_std, 2)
    
    # Unpack parameters more efficiently
    α_m = reshape(params[1:4*n_asvab], n_asvab, 4)
    γ = params[4*n_asvab+1:5*n_asvab]
    σ_m = exp.(params[5*n_asvab+1:6*n_asvab])
    β = params[6*n_asvab+1:6*n_asvab+7]
    δ = params[6*n_asvab+8]
    σ_w = exp(params[6*n_asvab+9])
    
    # Pre-calculate wage means
    wage_means = X * β
    
    loglik = 0.0
    
    # Parallelize over observations if sample is large enough
    for i in 1:n_obs
        obs_lik = 0.0
        for (k, (node, weight)) in enumerate(zip(nodes, weights))
            ξ = node
            
            # Calculate measurement likelihood
            meas_lik = calculate_measurement_likelihood(X_m[i,:], M_std[i,:], α_m, γ, σ_m, ξ)
            
            # Calculate wage likelihood
            μ_w = wage_means[i] + δ*ξ
            wage_lik = pdf(Normal(μ_w, σ_w), y[i])
            
            obs_lik += weight * meas_lik * wage_lik
        end
        loglik += log(max(obs_lik, 1e-10))
    end
    
    return -loglik
end

FA = fit(FactorAnalysis, transpose(M_std); maxoutdim=1)
fa_loadings = vec(loadings(FA))
fa_scores = vec(transpose(MultivariateStats.transform(FA, transpose(M_std))))

initial_wage_model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), data_sample)
initial_β = coef(initial_wage_model)

n_asvab = length(asvab_vars)
initial_params = vcat(
    zeros(4*n_asvab),     # α_m - start at zero
    fa_loadings,          # γ - use FA loadings
    fill(-1.0, n_asvab),  # log(σ_m) - start at small positive values
    initial_β,            # β - use regression results
    1.0,                  # δ - start at 1
    0.0                   # log(σ_w) - start at 1
)

println("Starting optimization with sample size: ", n_sample)
println("This should take about 5-10 minutes...")

result = optimize(
    params -> log_likelihood(params, M_std, X_m, X, data_sample.logwage, quad_nodes, quad_weights),
    initial_params,
    NelderMead(),  # More robust than BFGS for this problem
    Optim.Options(
        show_trace=true,
        iterations=200,    # Fewer iterations
        time_limit=300.0  # 5 minute time limit
    )
)

params_mle = Optim.minimizer(result)
println("\nOptimization finished!")
println("Convergence: ", Optim.converged(result))
println("Final value: ", -Optim.minimum(result))


println("\nMLE Results (based on sample of ", n_sample, " observations):")
println("\nMeasurement Equation Results:")
for j in 1:n_asvab
    println("\nASVAB ", asvab_vars[j])
    α_j = params_mle[((j-1)*4+1):(j*4)]
    γ_j = params_mle[4*n_asvab+j]
    σ_j = exp(params_mle[5*n_asvab+j])
    println("α (intercept, black, hispanic, female): ", round.(α_j, digits=3))
    println("γ (factor loading): ", round(γ_j, digits=3))
    println("σ (measurement error): ", round(σ_j, digits=3))
end

println("\nWage Equation Results:")
β = params_mle[6*n_asvab+1:6*n_asvab+7]
δ = params_mle[6*n_asvab+8]
σ_w = exp(params_mle[6*n_asvab+9])
println("β coefficients:")
vars = ["constant", "black", "hispanic", "female", "schoolt", "gradHS", "grad4yr"]
for (var, coef) in zip(vars, β)
    println(rpad(var, 10), ": ", round(coef, digits=3))
end
println("δ (factor effect): ", round(δ, digits=3))
println("σ (wage error): ", round(σ_w, digits=3))

# Question 7

using Test
using DataFrames, CSV, GLM, Statistics, DataFramesMeta
using HTTP, LinearAlgebra, Random, MultivariateStats, Optim
using Distributions, FastGaussQuadrature

@testset "Problem Set 8 Tests" begin
    # Sample data for testing
    test_data = DataFrame(
        wage = [50000.0, 45000.0, 60000.0],
        black = [0, 1, 0],
        hispanic = [1, 0, 0],
        female = [1, 1, 0],
        schoolt = [12, 16, 14],
        gradHS = [1, 1, 1],
        grad4yr = [0, 1, 0],
        asvabAR = [70.0, 80.0, 90.0],
        asvabCS = [75.0, 85.0, 95.0],
        asvabMK = [72.0, 82.0, 92.0],
        asvabNO = [73.0, 83.0, 93.0],
        asvabPC = [74.0, 84.0, 94.0],
        asvabWK = [76.0, 86.0, 96.0]
    )
    test_data.logwage = log.(test_data.wage)

    @testset "Question 1: Basic Regression" begin
        # Test basic regression model
        model1 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), test_data)
        
        @test model1 isa StatsModels.TableRegressionModel
        @test nobs(model1) == 3
        @test coefnames(model1) == ["(Intercept)", "black", "hispanic", "female", "schoolt", "gradHS", "grad4yr"]
    end

    @testset "Question 2: ASVAB Correlations" begin
        # Test correlation matrix calculation
        asvab_vars = ["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]
        asvab_matrix = Matrix(test_data[:, asvab_vars])
        cor_matrix = cor(asvab_matrix)
        
        @test size(cor_matrix) == (6, 6)
        @test all(diag(cor_matrix) .≈ 1.0)
        @test issymmetric(cor_matrix)
        @test all(-1.0 .<= cor_matrix .<= 1.0)
    end

    @testset "Question 3: Full ASVAB Regression" begin
        # Test regression with all ASVAB variables
        model3 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
                           asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), test_data)
        
        @test model3 isa StatsModels.TableRegressionModel
        @test nobs(model3) == 3
        @test length(coef(model3)) == 13  # intercept + 12 variables
    end

    @testset "Question 4: PCA" begin
        # Test PCA implementation
        asvab_vars = ["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]
        asvab_matrix = Matrix(test_data[:, asvab_vars])
        asvab_std = (asvab_matrix .- mean(asvab_matrix, dims=1)) ./ std(asvab_matrix, dims=1)
        M = fit(PCA, transpose(asvab_std); maxoutdim=1)
        pc1 = MultivariateStats.transform(M, transpose(asvab_std))
        
        @test size(pc1, 1) == 1  # One principal component
        @test size(pc1, 2) == 3  # Three observations
        @test principalratio(M) > 0 && principalratio(M) ≤ 1
    end

    @testset "Question 5: Factor Analysis" begin
        # Test Factor Analysis implementation
        asvab_vars = ["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK"]
        asvab_matrix = Matrix(test_data[:, asvab_vars])
        asvab_std = (asvab_matrix .- mean(asvab_matrix, dims=1)) ./ std(asvab_matrix, dims=1)
        FA = fit(FactorAnalysis, transpose(asvab_std); maxoutdim=1)
        factor_scores = MultivariateStats.transform(FA, transpose(asvab_std))
        
        @test size(factor_scores, 1) == 1  # One factor
        @test size(factor_scores, 2) == 3  # Three observations
        @test size(loadings(FA), 1) == 6  # Six ASVAB tests
    end

    @testset "Question 6: Measurement System" begin
        # Test matrix creation function
        function create_X_matrices(data)
            X_m = hcat(ones(nrow(data)), 
                      Float64.(data.black), 
                      Float64.(data.hispanic), 
                      Float64.(data.female))
            
            X = hcat(ones(nrow(data)), 
                     Float64.(data.black), 
                     Float64.(data.hispanic), 
                     Float64.(data.female),
                     Float64.(data.schoolt),
                     Float64.(data.gradHS),
                     Float64.(data.grad4yr))
            
            return X_m, X
        end

        X_m, X = create_X_matrices(test_data)
        @test size(X_m) == (3, 4)  # 3 observations, 4 variables
        @test size(X) == (3, 7)    # 3 observations, 7 variables
        
        # Test likelihood function inputs
        n_quad = 5
        nodes, weights = gausslegendre(n_quad)
        quad_nodes = √2 .* nodes
        quad_weights = weights ./ √π
        
        @test length(quad_nodes) == n_quad
        @test length(quad_weights) == n_quad
        @test sum(quad_weights) ≈ 1.0 rtol=1e-10
        
        # Test parameter vector structure
        n_asvab = 6
        initial_params = vcat(
            zeros(4*n_asvab),  # α_m
            ones(n_asvab),     # γ
            zeros(n_asvab),    # log(σ_m)
            zeros(7),          # β
            1.0,               # δ
            0.0                # log(σ_w)
        )
        
        @test length(initial_params) == 4*n_asvab + n_asvab + n_asvab + 7 + 1 + 1
    end
end