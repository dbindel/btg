using LinearAlgebra

include("./kernel.jl")
include("./transform.jl")
include("./incremental.jl")
include("./kernel_system.jl")

@doc raw"""
    Data

A struct containing at most `n` data points stored in column major order. `X` contains data locations, `F` contains data covariates, `Y` contains observations. The data struct may store at most `capacity` points.
"""
mutable struct Data
    capacity::Int
    n::Int
    x::IncrementalColumns{Float64} # Data locations
    fx::IncrementalColumns{Float64} # Covariates
    y::IncrementalVector{Float64} # Observations
end
function Data(capacity, x, fx, y)
    @assert size(x, 2) == size(fx, 2) == size(y, 1)
    n = size(x, 2)
    new_x = incremental_columns(capacity, x)
    new_fx = incremental_columns(capacity, fx)
    new_y = incremental_vector(capacity, y)
    return Data(capacity, n, new_x, new_fx, new_y)
end
function Data(x, fx, y)
    n = size(x, 2)
    return Data(n, x, fx, y)
end
function Data(capacity, d::Int, p::Int)
    new_x = incremental_columns(Float64, capacity, d)
    new_fx = incremental_columns(Float64, capacity, p)
    new_y = incremental_vector(Float64, capacity)
    return Data(capacity, 0, new_x, new_fx, new_y)
end
function add_point!(d::Data, x, fx, y)
    @assert d.n < d.capacity
    add_col!(d.x, x)
    add_col!(d.fx, fx)
    add_element!(d.y, y)
    d.n += 1
    return nothing
end

@doc raw"""
"""
mutable struct ComputeData
    capacity::Int
    n::Int
    d1::Data # unisolvent points
    d2::Data # Reduced dataset - see sjtu notes
    f1::LU{Float64, Matrix{Float64}} # LU factorization of covariates
    w::IncrementalColumns{Float64}
end
function ComputeData(capacity, d1, x, fx, y)
    d2 = Data(capacity, x, fx, y)
    f1 = lu(d1.fx)
    w = incremental_columns(capacity, f1 \ fx)
    return ComputeData(capacity, 0, d1, d2, f1, w)
end

function add_point!(comp::ComputeData, x, fx, y)
    @assert comp.n < comp.capacity
    comp.n += 1
    add_point!(comp.d2, x, fx, y)
    add_col!(comp.w, comp.f1 \ fx) # Add new row to W
    return nothing
end

mutable struct ComputeKernel{K<:AbstractCorrelation,T}
    capacity::Int
    n::Int
    k::K
    θ::T
    k1::Matrix{Float64}
    k12::IncrementalColumns{Float64}
    k2_tilde::IncrementalCholesky{Float64}
end
function ComputeKernel(capacity, comp::ComputeData, k, θ, k0)
    p = size(d1.fx, 1)
    k1 = Array{Float64}(undef, p, p)
    pairwise!(k1, k, d1.x, d1.x, θ...)
    k12 = Array{Float64}(undef, p, capacity)
    k2 = Array{Float64}(undef, capacity, capacity)
    return ComputeKernel(capacity, 0, k, θ, k1, k12, k2)
end

function add_point(c::ComputeKernel, comp::ComputeData, x)
    n = comp.n
    colwise!(c.k12[:, n], c.k, comp.d1.x[:, 1:comp.d1.n], x, c.θ...)
    # Update K2
    colwise!(c.k2[1:n, n], c.k, comp.d2.x[:, 1:comp.d2.n], x, c.θ...)
    # Update Ktilde
    tmp = c.k1 * comp.w[:, n]
    tmp2 = c.k12[:, 1:n]' * comp.w[:, n]
    c.k2[n, n] +=  dot(comp.w[:, n], tmp) - 2 * tmp2[n]
    c.k2[1:n-1, n] .+= (comp.w[:, 1:n-1]' * tmp) .- tmp2[1:n-1]
    # Update Cholesky factor of Ktilde
    # TODO write this more elegantly, this is messy
    v
    add_col!(c.k2_tilde, v)
    return nothing
end

@doc """
"""
struct BTG{G<:AbstractTransform,K<:AbstractCorrelation,C}
    horizon::Int
    n::Int
    g::G
    k::K
    comp::ComputeData
    comp_k::Vector{C}
end
function BTG()
end
