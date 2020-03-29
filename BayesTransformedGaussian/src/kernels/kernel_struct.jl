using Distances
using LinearAlgebra
include("../tools/plotting.jl")
include("../computation/derivatives.jl")
include("../computation/finitedifference.jl")


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
function correlation(k::AbstractCorrelation, x, θ...; jitter = 0, dims=1) #convention 
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
distance(k::FixedParam) = distance(k.k, k.theta...)

@doc raw"""
    Gaussian
"""
struct Gaussian <: AbstractCorrelation end
const RBF = Gaussian
const SqExponential = RBF

(::Gaussian)(τ, θ::Real) = exp(- τ * θ / 2)
(::Gaussian)(τ, ::AbstractVector) = exp(- τ / 2)
distance(::Gaussian, ::Real) = SqEuclidean()
distance(::Gaussian, θ::AbstractVector) = WeightedSqEuclidean(θ)

partial_θ(k::Gaussian, τ, θ) = -τ/2 * k(τ, θ) 
#partial_τ(k::Gaussian, τ, θ) = - θ/2 * k(τ, θ) 
#partial_θθ(k::RBF, τ, θ) = τ^2/4 * k(τ, θ)
#partial_ττ(k::Gaussian, τ, θ) =  θ^2/4 * k(τ, θ) 
 
struct Spherical <: AbstractCorrelation end
# TODO

struct Matern <: AbstractCorrelation end
# TODO

struct RationalQuadratic <: AbstractCorrelation end
# TODO


a = (tau, x) -> exp.(-x*tau/2)
da = (tau, x) -> -tau/2 * exp.(-x*tau/2)
gg = x -> a(2, x)
dgg =  x -> da(2, x)
(h, A, plt1, poly, fis1) = checkDerivative(gg, dgg, 1.0, nothing, 1, 2, 10)
#println(poly)
#display(plt1)

#dsdf = RBF()
dsdf = Gaussian()
ff = y -> dsdf(2, y)
dff =  y -> partial_θ(dsdf, 2, y)
(h, A, plt1, poly, fis2, vals, debugvals) = checkDerivative(ff, dff, 1.0, nothing, 1, 2, 10)
println(poly)
#display(plt1)

Plots.plot(vals[:, 1], seriestype = :scatter)
for i =2:10
    println("here")
    Plots.plot!(vals[:, i], seriestype = :scatter)
end

#Plots.plot(fis1')
#Plots.plot(fis2')

#plt(ff, .01, 10)
#plt!(dff, .01, 10)
#plt!(gg, .01, 10)
#plt!(dgg, .01, 10)

