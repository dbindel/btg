using LinearAlgebra

include("./kernels/kernel_struct.jl")
include("./transforms_struct.jl")

@doc raw"""
    Data

A struct containing at most `n` data points stored in column major order. `X` contains data locations, `F` contains data covariates, `Y` contains observations. The data struct may store at most `capacity` points.
"""
mutable struct Data
    capacity::Int
    n::Int
    x::Matrix{Float64} # Data locations
    fx::Matrix{Float64} # Covariates
    y::Vector{Float64} # Observations
end
function Data(capacity, x, fx, y)
    @assert size(x, 2) == size(fx, 2) == size(y, 1)
    n = size(x, 2)
    new_x = similar(x, size(x, 1), capacity)
    @views new_x[:, 1:n] .= x
    new_fx = similar(fx, size(fx, 1), capacity)
    @views new_fx[:, 1:n] .= fx
    new_y = similar(y, capacity)
    @views new_y[1:n] .= y
    return Data(capacity, n, new_x, new_fx, new_y)
end
function Data(x, fx, y)
    @assert size(x, 2) == size(fx, 2) == size(y, 1)
    n = size(x, 2)
    return Data(n, n, x, fx, y)
end
function Data(capacity, d::Int, p::Int)
    new_x = Array{Float64}(undef, d, capacity)
    new_fx = Array{Float64}(undef, p, capacity)
    new_y = Array{Float64}(undef, capacity)
    return Data(capacity, 0, new_x, new_fx, new_y)
end
function add_point(d::Data, x, fx, y)
    @assert d.n < d.capacity
    d.n += 1
    n = d.n
    d.x[:, n] .= x
    d.fx[:, n] .= fx
    d.y[n] = y
    return nothing
end

@doc raw"""
"""
mutable struct ComputeData
    capacity::Int
    n::Int
    d1::Data # unisolvent point
    d2::Data # Reduced dataset - see sjtu notes
    f0::LU{Float64, Matrix{Float64}} # LU factorization of covariates
    w::Matrix{Float64}
end
function ComputeData(capacity, d1)
    p = size(d1.fx, 1)
    d = size(d1.x, 1)
    d2 = Data(capacity, d, p)
    f0 = lu(d1.fx)
    w = Array{Float64}(undef, p, capacity)
    return ComputeData(capacity, 0, d1, d2, f0, w)
end

function add_point(comp::ComputeData, x, fx, y)
    @assert comp.n < comp.capacity
    comp.n += 1
    add_point(comp.d2, x, fx, y)
    ldiv!(view(comp.w, :, comp.n), comp.f0, fx) # Add new row to W
    return nothing
end

mutable struct ComputeKernel{K<:AbstractCorrelation,T}
    capacity::Int
    n::Int
    k::K
    θ::T
    k1::Matrix{Float64}
    k12::Matrix{Float64}
    k2::Matrix{Float64}
end
function ComputeKernel(capacity, comp::ComputeData, k, θ)
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
    if n == 1
        # begin cholesky factorization of Ktilde
        c.k2[1, 1] = c.k(x, x, c.θ...)
        tmp = c.k1 * comp.w[:, 1]
        tmp2 = c.k12[:, 1]' * comp.w[:, 1]
        c.k2[1, 1] += dot(comp.w[:, 1], tmp) - 2 * tmp2
        c.k2[1, 1] = sqrt(c.k2[1, 1])
    else
        # Update K2
        colwise!(c.k2[1:n, n], c.k, comp.d2.x[:, 1:comp.d2.n], x, c.θ...)
        # Update Ktilde
        tmp = c.k1 * comp.w[:, n]
        tmp2 = c.k12[:, 1:n]' * comp.w[:, n]
        c.k2[n, n] +=  dot(comp.w[:, n], tmp) - 2 * tmp2[n]
        c.k2[1:n-1, n] .+= (comp.w[:, 1:n-1]' * tmp) .- tmp2[1:n-1]
        # Update Cholesky factor of Ktilde
        # TODO write this more elegantly, this is messy
        t = UpperTriangular(c.k2[1:n-1, 1:n-1])
        tmp3 = t \ c.k2[1:n-1, n]
        tmp4 = t' \ tmp3
        c.k2[1:n-1, n] .= tmp4
        @views c.k2[n, n] = sqrt(c.k2[n, n] - dot(c.k2[1:n-1, n], c.k2[1:n-1, n]))
    end
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
