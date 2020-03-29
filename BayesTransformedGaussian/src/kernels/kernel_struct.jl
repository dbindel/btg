using Distances
using LinearAlgebra

@doc raw"""
    AbstractCorrelation

An abstract type for all correlation functions.
"""
abstract type AbstractCorrelation end

(k::AbstractCorrelation)(x, y, θ...) = k(distance(k, θ...)(x, y))

"""
jitter: nugget term added to diagonal of kernel matrix to ensure positive-definiteness
dims: 1 if data points are arranged row-wise and 2 if col-wise
"""
function correlation(k::AbstractCorrelation, x, θ...; jitter = 0; dims=1) #convention 
    ret = Array{Float64}(undef, size(x, dims), size(x, dims))
    correlation!(ret, k, x, θ...; jitter = jitter)
    return ret
end
function cross_correlation(k::AbstractCorrelation, x, y, θ...)
    ret = Array{Float64}(undef, size(x, 2), size(y, 2))
    cross_correlation!(ret, k, x, y, θ...)
    return ret
end

function cross_correlation!(out, k::AbstractCorrelation, x, y, θ...; dims = 1)
    dist = distance(k, θ...)
    pairwise!(out, dist, x, y, dims=dims)
    out .= (τ -> k(τ, θ...)).(out)
    return nothing
end
function correlation!(out, k::AbstractCorrelation, x, θ...; jitter = 0, dims=1)
    dist = distance(k, θ...)
    pairwise!(out, dist, x, dims=dims)
    out .= (τ -> k(τ, θ...)).(out)
    if jitter != 0
        out +=UniformScaling(jitter) 
        out ./= out[1, 1] #covariance must be in [0, 1]
    end
    return nothing
end

@doc raw"""
"""
struct FixedParam{K<:AbstractCorrelation,T} <: AbstractCorrelation
    k::K
    θ::T
end
FixedParam(k, θ...) = FixedParam(k, θ)

(k::FixedParam)(τ) = k.k(τ, k.θ...)
distance(k::FixedParam, θ...) = distance(k.k, θ...)

@doc raw"""
    Gaussian
"""
struct Gaussian <: AbstractCorrelation end
const RBF = Gaussian
const SqExponential = RBF

(::Gaussian)(τ, ℓ::Real) = exp(- τ / 2 / ℓ)
(::Gaussian)(τ, ::AbstractVector) = exp(- τ / 2)
distance(::Gaussian, ::Real) = SqEuclidean()
distance(::Gaussian, ℓ::AbstractVector) = WeightedSqEuclidean(ℓ)

partial_θ(k::Gaussian, τ, ℓ) = τ * k(τ, ℓ) / 2 / ℓ ^ 2
partial_τ(k::Gaussian, τ, ℓ) = - k(τ, ℓ) / 2 / ℓ
partial_θθ(k::RBF, τ, ℓ) = τ * k(τ, ℓ) * (τ - 4 * ℓ) / 4 / ℓ ^ 4

struct Spherical <: AbstractCorrelation end
# TODO

struct Matern <: AbstractCorrelation end
# TODO

struct RationalQuadratic <: AbstractCorrelation end
# TODO
