@doc raw"""
"""
abstract type RadialCorrelation end

@doc raw"""
"""
struct ExponentiatedQuadratic <: Correlation end
const Gaussian = ExponentiatedQuadratic
const RBF = ExponentiatedQuadratic
const SquaredExponential = ExponentiatedQuadratic
const SqExponential = ExponentiatedQuadratic

(::RBF)(θ, τ) = exp(-τ / 2)

_dist_type(ℓ, x, y) = promote_type(eltype(ℓ), eltype(x), eltype(y))

struct Spherical <: Correlation end
# TODO

struct Matern <: Correlation end
# TODO

struct RationalQuadratic <: Correlation end
# TODO

@doc raw"""
"""
abstract type LengthScale end
function (d::LengthScale)(ℓ, x, y)
    out = Array{_dist_type(ℓ, x, y)}(undef, size(x, 2), size(y, 2))
    d(out, ℓ, x, y)
    return out
end

@doc raw"""
"""
struct Scale <: LengthScale end
function (d!::Scale)(out, ℓ, x, y)
    pairwise!(out, SqEuclidean(), x, y, dims=2)
    out ./= ℓ
    return nothing
end

@doc raw"""
"""
struct MultiScale <: LengthScale end
function (d!::MultiScale)(out, ℓ, x, y)
    pairwise!(out, WeightedSqEuclidean(ℓ), x, y, dims=2)
    return nothing
end
