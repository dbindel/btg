using Distances
using LinearAlgebra
include("../tools/plotting.jl")
#include("../computation/derivatives.jl")
include("../computation/finitedifference.jl")
#
#CONVENTIONS:
#
# locations x and y will always be arrays, whether it's 1D or 2D
#
#
# NOTES:
# - WeightedSqEuclidean([3, 4])(x, y) from Distances.jl will default to SqEuclidean if x and y aren't arrays
#   it will throw an error of x and y are arrays but not the same length as the weights vector
# - one can input single length scale as scalar or array of length 1
#   and multiple length scale as array
#
#
@doc raw"""
    AbstractCorrelation

An abstract type for all correlation functions.
"""
abstract type AbstractCorrelation end

(k::AbstractCorrelation)(x::Array, y::Array, θ) = k(distance(k, θ)(x, y),  θ)

"""
Computes matrix of pairwise distances between x and y, which contains points 
arranged row-wise
"""
function distancematrix(k::AbstractCorrelation, θ, x, y=x; dims=1)
    ret = Array{Float64}(undef, size(x, dims), size(x, dims))
    computeDists!(ret, k, θ, x, y; dims=dims)
    return ret
end

"""
x, y: arrays of points in R^d
θ: 1D array
"""
function computeDists!(out, k::AbstractCorrelation, θ, x, y=x; dims=1)
    if typeof(θ)<:Array
        @assert max(size(θ, 1), size(θ, 2)) == size(x, 2) == size(y, 2)
    end
    dist = distance(k, θ)
    try
        out .= pairwise!(out, dist, x, y, dims = dims)
    catch(MethodError)
        out .= pairwise!(out, dist, reshape(x,size(x, 1), size(x, 2)), reshape(y, size(y, 1), size(y, 2)), dims = dims)
    end   
        return nothing
end


"""
jitter: nugget term added to diagonal of kernel matrix to ensure positive-definiteness
dims: 1 if data points are arranged row-wise and 2 if col-wise

"""
function correlation(k::AbstractCorrelation, x, θ; jitter = 0, dims=1) 
    ret = Array{Float64}(undef, size(x, dims), size(x, dims))
    correlation!(ret, k, x, θ...; jitter = jitter)
    return ret
end

function cross_correlation(k::AbstractCorrelation, x, y, θ)
    ret = Array{Float64}(undef, size(x, 2), size(y, 2))
    cross_correlation!(ret, k, x, y, θ...)
    return ret
end

function cross_correlation!(out, k::AbstractCorrelation, x, y, θ; dims = 1)
    dist = distance(k, θ...)
    pairwise!(out, dist, x, y, dims=dims)
    out .= (τ -> k(τ, θ...)).(out)
    return nothing
end

function correlation!(out, k::AbstractCorrelation, x, θ; jitter = 0, dims=1)
    dist = distance(k, θ)
    pairwise!(out, dist, x, dims=dims)
    out .= (τ -> k(τ, θ)).(out)
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
distance(k::FixedParam) = distance(k.k, k.theta...)

@doc raw"""
    Gaussian
"""
struct Gaussian <: AbstractCorrelation end
const RBF = Gaussian
const SqExponential = RBF

#these two functions are purely meant to be used in conjunction with (k::AbstractCorrelation)(x::Array, y::Array, θ)
(::Gaussian)(τ::Real, θ::Real) = exp(- τ * θ / 2) 
(::Gaussian)(τ::Real, ::AbstractVector) = exp(- τ / 2) #theta already taken into account in computation of tau

distance(::Gaussian, θ::Real) = SqEuclidean(θ)
distance(::Gaussian, θ::AbstractVector) = max(size(θ, 1), size(θ, 2))==1 ? SqEuclidean(θ[1]) : WeightedSqEuclidean(θ)

"""
currently derivative is only implemented for single length scale
"""
function partial_θ(k::Gaussian, τ, θ)
    -τ/2 * k(τ, θ[1])
end
#partial_θ(k::Gaussian, τ, θ...) = ; 
#partial_τ(k::Gaussian, τ, θ) = - θ/2 * k(τ, θ) 
#partial_θθ(k::RBF, τ, θ) = τ^2/4 * k(τ, θ)
#partial_ττ(k::Gaussian, τ, θ) =  θ^2/4 * k(τ, θ) 
 
struct Spherical <: AbstractCorrelation end
# TODO

struct Matern <: AbstractCorrelation end
# TODO

struct RationalQuadratic <: AbstractCorrelation end
# TODO


