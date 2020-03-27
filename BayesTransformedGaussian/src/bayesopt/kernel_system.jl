include("./incremental.jl")
include("./kernel.jl")

@doc raw"""
"""
mutable struct KernelData{K<:AbstractCorrelation}
    # upper bound on the n - p samples
    capacity::Int
    n::Int
    # Kernel Correlation Function
    k::K
    # we pick a subset (p x p) subset of the data whose covariates are unisolvent
    # use reduced dataset to simplify computations - see sjtu notes
    # Fixed Size
    x1::Matrix{Float64} # p observation locations w/ unisolvent covariates
    fx1::LU{Float64, Matrix{Float64}} # Factorization of (p x p) covariate matrix
    K1::Matrix{Float64} # correlation of observation locations
    # Incremental
    x2::DataMatrix{Float64} # n - p observation locations
    W::DataMatrix{Float64} # fx1 \ fx2
    K12::DataMatrix{Float64} # Cross correlation of observation locations
    K2_tilde::IncrementalCholesky{Float64} # Reduced kernel system for x2
end
function kernel_data(capacity, k, x1, fx1, x2, fx2)
    @assert size(fx1, 1) == size(fx2, 1) == size(x1, 2)
    p = size(fx1, 1) # Check sizes
    n = size(x2, 2)
    
    fx1_fact = lu(fx1) # Factorize Fx
    K1 = Array{Float64}(undef, p, p)
    pairwise!(K1, k, x1) # Compute correlation of x1 locations

    new_x2 = data_array(capacity, x2)
    
    W = data_array(capacity, fx1_fact \ fx2) # Compute Fx1 \ Fx2
    WK1W = W' * (K1 * W)
    K12 = data_array(Float64, capacity, p)
    vw = extend!(K12, n)
    pairwise!(vw, k, x1, x2) # Compute cross correlations between locations
    K21W = K12' * W
    
    K22 = Array{Float64}(undef, n, n)
    pairwise!(K22, k, x2)
    # Compute reduced system
    K2_tilde = incremental_cholesky(capacity, K22 .- K21W .- K21W' .+ WK1W)

    return KernelData(capacity, n, k, x1, fx1_fact, K1, new_x2, W, new_K12, K2_tilde)
end

solve_lower(kd, y1, y2) = kd.K2_tilde \ (y2 .- (kd.W' * y1))
solve_upper(kd, c2) = -(kd.W * c2)
solve_tail(kd, y1, c1, c2) = kd.fx1' \ (y1 .- (kd.K1 * c1) .- (kd.K12 * c2))
function solve_system(kd, y1, y2) =
    c2 = solve_lower(kd, y1, y2)
    c1 = solve_upper(kd, c2)
    d = solve_tail(kd, y1, c1, c2)
    return c1, c2, d
end

function point_correlation!(k1, k2, kd, x)
    colwise!(k1, kd.x1, x)
    colwise!(k2, kd.x2, x)
    return nothing
end

function k2_update!(k2, kd)
    tmp .+= kd.W * (kd.K1 * kd.W[:, end])
    tmp .-= kd.K12' * kd.W[:, end]
    tmp[end] .-= dot(kd.K12[:, end], kd.W[:, end])
    return nothing
end

function add_point!(kd, x, fx)
    @assert kd.n + 1 <= kd.capacity
    n = kd.n
    
    add_point!(kd.x2, x)
    add_point!(kd.W, kd.fx1 \ fx)
    vw = extend!(kd.K12, 1)

    tmp = Array{Float64}(undef, n + 1)
    point_correlation!(vw, tmp, k2, kd, x)

    tmp .+= kd.W * (kd.K1 * kd.W[:, end])
    tmp .-= kd.K12' * kd.W[:, end]
    tmp[end] .-= dot(kd.K12[:, end], kd.W[:, end])

    add_col!(kd.K2_tilde, tmp)

    kd.n += 1
    
    return nothing
end

function remove_point!(kd)
    @assert kd.n - 1 >= 1
    remove_point!(kd.x2)
    remove_point!(kd.W)
    remove_point!(kd.K12)
    remove_col!(kd.K2_tilde)
    kd.n -= 1
    return nothing
end

function predict_point(kd::KernelSystem, y1, y2, x, fx)
    c1, c2, β = solve_system(kd, y1, y2)
    k1 = Array{Float64}(undef, size(kd.K1, 1))
    k2 = Array{Float64}(undef, kd.n)
    point_correlation!(k1, k2, kd, x)
    m = dot(k1, c1) + dot(k2, c2) + dot(fx, β)
    return m
end

function qmC!(kd::KernelSystem, y1, y2, x, fx)
    c1, c2, β = solve_system(kd, y1, y2)
    k1 = Array{Float64}(undef, size(kd.K1, 1))
    k2 = Array{Float64}(undef, kd.n)
    point_correlation!(k1, k2, kd, x)
    m = dot(k1, c1) + dot(k2, c2) + dot(fx, β)
    
    q = dot(y1, c1) + dot(y2, c2)
    
    add_point!(kd)
    C = kd.K2_tilde.R[kd.n, kd.n] ^ 2
    remove_point!(kd)
    
    return q, m, C
end
