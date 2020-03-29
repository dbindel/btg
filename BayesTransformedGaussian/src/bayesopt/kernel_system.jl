@doc raw"""
"""
mutable struct SystemData
    X::DataMatrix{Float64} # Observation locations
    U::LU{Float64, Matrix{Float64}} # p x p unisolvent subset of Fx
    W::DataMatrix{Float64} # U \ Fx (without the points in uFx)
end

function usage(ks::SystemData)
    return (p = size(ks.U, 1), n = size(ks.X, 2), capacity = ks.X.capacity)
end

@doc raw"""
"""
struct KernelData{K<:AbstractCorrelation}
    k::K # Kernel Correlation function
    K1::Matrix{Float64} # Correlation at locations for U
    K12::DataMatrix{Float64} # Cross correlation of observation locations
    rK2::IncrementalCholesky{Float64} # Reduced kernel system for x2
end

function kernel_data(capacity, k, X, W, θ...)
    return kernel_data(capacity, FixedParam(k, θ), X, W)
end

function kernel_data(capacity, k, X, W)
    p, n = size(W)
    @views X1, X2 = X[:, 1:p], X[:, p+1:end]

    K1 = Array{Float64}(undef, p, p)
    @views pairwise!(K1, k, X1, θ...)

    K12′ = Array{Float64}(undef, p, capacity)
    @views pairwise!(K12[:, 1:n],  k, X1, X2, θ...)

    rK2′ = Array{Float64}(undef, capacity, capacity)
    @views pairwise!(K2′[1:n, 1:n], k, X2, θ...)
    mul!(K2, W', K12, -1, 1)
    mul!(K2, K12', W, -1, 1)
    tmp = K1 * W
    mul!(K2, W', K1, 1, 1)

    return KernelData(k, K1, K12, rK2)
end

function usage(kd::KernelData)
    return (p = size(kd.K12, 1), n = size(kd.K12, 2), capacity = kd.K12.capacity)
end

log(ks::SystemData, kd::KernelData) = exp(logdet(ks, kd))
logdet(ks::KernelSystem, kd::KernelData) = 2 * logabsdet(ks.U)[1] + logdet(kd.rK2)

function kernel_data(capacity, p, k, X, W)
    m = size(X, 2) - p
    
    K1 = Array{Float64}(undef, p, p)
    @views pairwise!(K1, k, X[:, 1:p])
    K12 = data_array(Float64, capacity, p)
    vw = extend!(K12, n)
    @views pairwise!(vw, k, X[:, 1:p], X[:, p+1:end])
    K2 = Array{Float64}(undef, m, m)
    @views pairwise!(K2, k, X[:, p+1:end])
    
    WK1W = W' * (K1 * W)
    K21W = K12' * W

    K2_tilde = incremental_cholesky(capacity, K22 .- K21W .- K21W' .+ WK1W)

    return KernelData(p, m, capacity, K1, K12, K2_tilde)
end

function kernel_system(capacity, k, X, Fx)
    p, n = size(X)
end

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

    return KernelData(capacity, n, k, x1, fx1_fact, K1, new_x2, W, K12, K2_tilde)
end

solve_lower(kd, y1, y2) = kd.K2_tilde \ (y2 .- (kd.W' * y1))
solve_upper(kd, c2) = -(kd.W * c2)
solve_tail(kd, y1, c1, c2) = kd.fx1' \ (y1 .- (kd.K1 * c1) .- (kd.K12 * c2))
function solve_system(kd, y1, y2)
    c2 = solve_lower(kd, y1, y2)
    c1 = solve_upper(kd, c2)
    d = solve_tail(kd, y1, c1, c2)
    return c1, c2, d
end

function point_correlation!(k1, k2, kd, x)
    colwise!(k1, kd.k, kd.x1, x)
    colwise!(k2, kd.k, kd.x2, x)
    return nothing
end

function k2_update!(k2, kd)
    k2 .+= kd.W' * (kd.K1 * kd.W[:, end])
    k2 .-= kd.K12' * kd.W[:, end] .+ kd.W' * kd.K12[:, end]
    return nothing
end

function add_point!(kd, x, fx)
    @assert kd.n + 1 <= kd.capacity
    n = kd.n
    
    add_point!(kd.x2, x)
    add_point!(kd.W, kd.fx1 \ fx)
    vw = extend!(kd.K12, 1)

    tmp = Array{Float64}(undef, n + 1)
    point_correlation!(vw, tmp, kd, x)
    k2_update!(tmp, kd)

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

function predict_point(kd::KernelData, y1, y2, x, fx)
    c1, c2, β = solve_system(kd, y1, y2)
    k1 = Array{Float64}(undef, size(kd.K1, 1))
    k2 = Array{Float64}(undef, kd.n)
    point_correlation!(k1, k2, kd, x)
    m = dot(k1, c1) + dot(k2, c2) + dot(fx, β)
    return m
end

function qmC!(kd::KernelData, y1, y2, x, fx)
    c1, c2, β = solve_system(kd, y1, y2)
    k1 = Array{Float64}(undef, size(kd.K1, 1))
    k2 = Array{Float64}(undef, kd.n)
    point_correlation!(k1, k2, kd, x)
    m = dot(k1, c1) + dot(k2, c2) + dot(fx, β)
    
    q = dot(y1, c1) + dot(y2, c2)
    
    add_point!(kd, x, fx)
    C = kd.K2_tilde.R[kd.n, kd.n] ^ 2
    remove_point!(kd)
    
    return q, m, C
end
