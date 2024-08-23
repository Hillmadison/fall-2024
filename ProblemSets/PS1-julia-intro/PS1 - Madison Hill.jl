# ECON 6343: Econometrics III - Problem Set 1
# Madison Hill

using JLD2, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions

# Question 1: Initializing variables and practice with basic matrix operations
function q1()
    Random.seed!(1234)
    
    # (a) Create matrices
    A = rand(Uniform(-5, 10), 10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    C = [A[1:5, 1:5] B[1:5, 6:7]]
    D = [a <= 0 ? a : 0 for a in A]
    
    # (b) Number of elements in A
    println("Number of elements in A: ", length(A))
    
    # (c) Number of unique elements in D
    println("Number of unique elements in D: ", length(unique(D)))
    
    # (d) Create E (vec of B)
    E = reshape(B, :, 1)
    
    # (e) Create 3D array F
    F = cat(A, B, dims=3)
    
    # (f) Twist F using permutedims
    F = permutedims(F, (2, 1, 3))
    
    # (g) Create G (Kronecker product of B and C)
    G = kron(B, C)
    
    # (h) Save matrices as .jld2 file
    jldsave("matrixpractice.jld2"; A, B, C, D, E, F, G)
    
    # (i) Save subset of matrices as .jld2 file
    jldsave("firstmatrix.jld2"; A, B, C, D)
    
    # (j) Export C as .csv file
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))
    
    # (k) Export D as tab-delimited .dat file
    CSV.write("Dmatrix.dat", DataFrame(D, :auto), delim='\t')
    
    return A, B, C, D
end

# Question 2: Practice with loops and comprehensions
function q2(A, B, C)
    # (a) Element-wise product of A and B
    AB = [A[i,j] * B[i,j] for i in 1:size(A,1), j in 1:size(A,2)]
    AB2 = A .* B

    # (b) Elements of C between -5 and 5
    Cprime = [c for c in C if -5 <= c <= 5]
    Cprime2 = filter(c -> -5 <= c <= 5, vec(C))

    # (c) Create 3D array X
    N, K, T = 15169, 6, 5
    X = Array{Float64}(undef, N, K, T)
    for t in 1:T
        X[:, 1, t] .= 1  # intercept
        X[:, 2, t] = rand(Bernoulli(0.75 * (6 - t) / 5), N)
        X[:, 3, t] = rand(Normal(15 + t - 1, 5 * (t - 1)), N)
        X[:, 4, t] = rand(Normal(π * (6 - t) / 3, 1/ℯ), N)
        X[:, 5, t] = rand(Binomial(20, 0.6), N)
        X[:, 6, t] = rand(Binomial(20, 0.5), N)
    end

    # (d) Create β matrix
    β = [
        [1 + 0.25*(t-1) for t in 1:T]';
        [log(t) for t in 1:T]';
        [-sqrt(t) for t in 1:T]';
        [exp(t) - exp(t+1) for t in 1:T]';
        [t for t in 1:T]';
        [t/3 for t in 1:T]'
    ]

    # (e) Create Y matrix
    Y = [sum(X[i,:,t] .* β[:,t]) + rand(Normal(0, 0.36)) for i in 1:N, t in 1:T]
end

# Question 3: Reading in Data and calculating summary statistics
function q3()
    # (a) Import and process data
    possible_paths = [
        "nlsw88.csv",  # Current directory
        "data/nlsw88.csv",  # In a data subdirectory
        "../data/nlsw88.csv",  # In a data directory one level up
        joinpath(dirname(@__FILE__), "nlsw88.csv"),  # Same directory as this script
        joinpath(dirname(@__FILE__), "data", "nlsw88.csv")  # In a data subdirectory relative to this script
    ]
    
    file_path = ""
    for path in possible_paths
        if isfile(path)
            file_path = path
            break
        end
    end
    
    if isempty(file_path)
        error("Could not find nlsw88.csv. Please ensure it's in the correct location.")
    else
        println("Found nlsw88.csv at: ", file_path)
    end
    
    df = CSV.read(file_path, DataFrame)
    println("Columns in the dataframe: ", names(df))
    df = dropmissing(df)  # Remove rows with missing values
    CSV.write("nlsw88_processed.csv", df)

    # (b) Percentage never married and college graduates
    pct_never_married = mean(df.never_married) * 100
    pct_college_grad = mean(df.collgrad) * 100
    println("Percentage never married: ", pct_never_married)
    println("Percentage college graduates: ", pct_college_grad)

    # (c) Percentage in each race category
    race_freq = freqtable(df, :race)
    race_pct = prop(race_freq) * 100
    println("Race percentages:")
    println(race_pct)

    # (d) Summary statistics
    summarystats = describe(df)
    println("Summary statistics:")
    println(summarystats)
    println("Missing grade observations: ", sum(ismissing.(df.grade)))

    # (e) Joint distribution of industry and occupation
    industry_occupation = freqtable(df, :industry, :occupation)
    println("Joint distribution of industry and occupation:")
    println(prop(industry_occupation) * 100)

    # (f) Mean wage over industry and occupation
    wage_by_ind_occ = combine(groupby(df, [:industry, :occupation]), :wage => mean)
    println("Mean wage by industry and occupation:")
    println(wage_by_ind_occ)

    return df  # Return the dataframe for use in other functions if needed
end

# Question 4: Practice with functions
function q4(df)
    println("Starting q4 function...")
    
    # (a) Load matrices
    println("Loading matrices from firstmatrix.jld2...")
    data = load("firstmatrix.jld2")
    A, B = data["A"], data["B"]
    println("Matrices A and B loaded. Sizes: ", size(A), " and ", size(B))

    # (b) & (c) Define matrixops function
    function matrixops(A, B)
        println("matrixops called with inputs of size ", size(A), " and ", size(B))
        if size(A) != size(B)
            error("inputs must have the same size")
        end
        return A .* B, A' * B, sum(A + B)
    end

    # (d) Evaluate matrixops with A and B
    println("Evaluating matrixops with A and B...")
    result_AB = matrixops(A, B)
    println("Result of matrixops(A, B):")
    println(result_AB)

    # (f) Evaluate matrixops with C and D
    println("Evaluating matrixops with C and D...")
    C, D = data["C"], data["D"]
    try
        result_CD = matrixops(C, D)
        println("Result of matrixops(C, D):")
        println(result_CD)
    catch e
        println("Error in matrixops(C, D): ", e)
        println("Sizes: C $(size(C)), D $(size(D))")
    end

    # (g) Evaluate matrixops with ttl_exp and wage from df
    println("Preparing ttl_exp and wage from dataframe...")
    if :ttl_exp in propertynames(df) && :wage in propertynames(df)
        ttl_exp = df.ttl_exp
        wage = df.wage
        println("Types: ttl_exp $(typeof(ttl_exp)), wage $(typeof(wage))")
        println("Lengths: ttl_exp $(length(ttl_exp)), wage $(length(wage))")
        
        # Convert to matrices
        ttl_exp_matrix = reshape(Vector{Float64}(ttl_exp), :, 1)
        wage_matrix = reshape(Vector{Float64}(wage), :, 1)
        println("Converted to matrices. Sizes: ttl_exp $(size(ttl_exp_matrix)), wage $(size(wage_matrix))")
        
        # Ensure both matrices have the same size
        min_length = min(size(ttl_exp_matrix, 1), size(wage_matrix, 1))
        ttl_exp_matrix = ttl_exp_matrix[1:min_length, :]
        wage_matrix = wage_matrix[1:min_length, :]
        
        println("Final matrix sizes: ttl_exp $(size(ttl_exp_matrix)), wage $(size(wage_matrix))")
        
        try
            println("Calling matrixops with ttl_exp and wage...")
            result_exp_wage = matrixops(ttl_exp_matrix, wage_matrix)
            println("Result of matrixops(ttl_exp, wage):")
            println(result_exp_wage)
        catch e
            println("Error in matrixops(ttl_exp, wage): ", e)
            println("Error type: ", typeof(e))
            println("Sizes: ttl_exp $(size(ttl_exp_matrix)), wage $(size(wage_matrix))")
        end
    else
        println("Error: ttl_exp or wage not found in the dataframe")
        println("Available columns: ", propertynames(df))
    end
end

# Main execution
println("Calling q1()...")
A, B, C, D = q1()
println("Calling q2()...")
q2(A, B, C)
println("Calling q3()...")
df = q3()  # Capture the returned dataframe
println("Calling q4()...")
q4(df)  # Pass df to q4

println("Script execution completed.")

# Question 5: Unit tests
# Note: Typically, unit tests would be in a separate file. 
# Here's a basic example of how you might structure them:

using Test

@testset "Problem Set 1 Tests" begin
    @testset "q1 Tests" begin
        A, B, C, D = q1()
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)
        @test size(D) == (10, 7)
        @test all(D .<= 0)
    end

    @testset "q2 Tests" begin
        # Add tests for q2 function
    end

    @testset "q3 Tests" begin
        # Add tests for q3 function
    end

    @testset "q4 Tests" begin
        # Add tests for q4 function and matrixops function
    end
end


