using Distances
import Distances: pairwise!

@doc raw"""
"""
abstract type Correlation end

@doc raw"""
"""
struct InducedQuadratic{K<:Correlation} <: Correlation
    k::K
end

(k::InducedQuadratic)(x, y, ℓ::Real, θ...) = k.k(sqeuclidean(x, y) / ℓ, θ...)
function (k::InducedQuadratic)(x, y, ℓ::AbstractVector, θ...)
    dist = WeightedSqEuclidean(inv.(ℓ))
    return k.k(dist(x, y), θ...)
end

function pairwise!(out, k::InducedQuadratic, x, y, ℓ::Real, θ...)
    pairwise!(out, SqEuclidean(), x, y, dims=2)
    out .= (τ -> k.k(τ / ℓ, θ...)).(out)
    return nothing
end
function pairwise!(out, k::InducedQuadratic, x, y, ℓ::AbstractVector, θ...)
    dist = WeightedSqEuclidean(inv.(ℓ))
    pairwise!(out, dist, x, y, dims=2)
    out .= (τ -> k.k(τ, θ...)).(out)
    return nothing
end


@doc raw"""
    """
struct ExponentiatedQuadratic <: Correlation end
@inline (::ExponentiatedQuadratic)(τ) = exp(-τ / 2)
(k::ExponentiatedQuadratic)(x, y) = k(sqeuclidean(x, y))
function pairwise!(out, k::ExponentiatedQuadratic, x, y)
    pairwise!(out, SqEuclidean(), x, y, dims=2)
    out .= k.(out)
    return nothing
end

const RBF = InducedQuadratic{ExponentiatedQuadratic}
const Gaussian = RBF
RBF() = InducedQuadratic(ExponentiatedQuadratic())

partial_θ(k::RBF, τ, ℓ) = τ * k(τ, ℓ) / 2 / ℓ ^ 2
function partial_θ!(out, k::RBF, x, y, ℓ)
    pairwise!(out, SqEuclidean(), x, y, dims=2)
    out .= (τ -> partial_θ(k, τ, ℓ)).(out)
    return nothing
end
partial_θθ(k::RBF, τ, ℓ) = τ * k(τ, ℓ) * (τ - 4 * ℓ) / 4 / ℓ ^ 4
function partial_θθ!(out, k::RBF, x, y, ℓ)
    pairwise!(out, SqEuclidean(), x, y, dims=2)
    out .= (τ -> partial_θθ(k, τ, ℓ)).(out)
    return nothing
end

struct Spherical <: Correlation end
# TODO

struct Matern <: Correlation end
# TODO

struct RationalQuadratic <: Correlation end
# TODO
