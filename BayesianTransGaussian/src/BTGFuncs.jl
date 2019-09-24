"""
    BTGFuncs

The `BTGFuncs` module provides transformation, covariate, and correlation 
functions used in BTG models.
"""
module BTGFuncs

using Distributions

export IsotropicCorrelation,
    ExpCorr,
    MaternCorr,
    RationalCorr,
    SphericalCorr,
    PowerTransform,
    BoxCox,
    DPowerTransform,
    DBoxCox,
    YeoJohnson,
    DYeoJohnson

"""
    IsotropicCorrelation{R<:AbstractFloat}

The `IsotropicCorrelation` type is intended to be inherited 
when defining functors representing parameterized isotropic
correlation functions.
"""
abstract type IsotropicCorrelation{R<:AbstractFloat} end

"""
TODO
"""
struct ExpCorr{R} <: IsotropicCorrelation{R}
    θ1::Uniform{R}
    θ2::Uniform{R}
end

function (k::ExpCorr)(θ1, θ2, τ)
    return θ1^(τ^θ2)
end

"""
TODO
"""
struct MaternCorr{R} <: IsotropicCorrelation{R}
    θ1::Uniform{R}
    θ2::Uniform{R}
end

function (k::MaternCorr)(θ1, θ2, τ) 
    # TODO
    return one(τ)
end

"""
TODO
"""
struct RationalCorr{R} <: IsotropicCorrelation{R}
    θ1::Uniform{R}
    θ2::Uniform{R}
end

function (k::RationalCorr)(θ1, θ2, τ)
    return if τ == zero(τ)
        one(τ)
    else
        θ1′ = -log(θ1)
        θ2′ = -log(θ2)
        τ′ = τ / θ1′
        (one(τ′) + τ′^2)^(-θ2′)
    end
end

"""
TODO
"""
struct SphericalCorr{R} <: IsotropicCorrelation{R}
    θ::Uniform{R}
end

function (k::SphericalCorr)(θ, τ)
    return if τ == zero(τ)
        one(τ)
    else
        θ′ = -log(θ)
        if τ <= θ′
            τ′ = τ / θ′
            one(τ′) - τ′ * (typeof(τ′, 3) - τ′^2) / typeof(τ′, 2)
        else
            zero(τ)
        end
    end
end

"""
    PowerTransform{R<:AbstractFloat}

The `PowerTransform` type is intended to be inherited 
when defining functors representing parameterized power
transformations.
"""
abstract type PowerTransform{R<:AbstractFloat} end

"""
TODO
"""
struct BoxCox{R} <: PowerTransform{R}
    λ::Uniform{R}
end

function (g::BoxCox)(λ, x)
    return λ == zero(λ) ? log(x) : expm1(log(x) * λ) / λ
end

"""
TODO
"""
struct YeoJohnson{R} <: PowerTransform{R}
    λ::Uniform{R}
end

function (g::YeoJohnson)(λ, x)
    return if λ != zero(λ) && x >= zero(x)
        expm1(λ * log1p(x)) / λ
    elseif λ == zero(λ) && x >= zero(x)
        log1p(x)
    elseif λ != oftype(λ, 2) && x < zero(x)
        -expm1((2 - λ) * log1p(-x)) / (2 - λ)
    else
        -log1p(-x)
    end
end

"""
    DPowerTransform{R<:AbstractFloat}

The `DPowerTransform` type is intended to be inherited 
when defining functors representing derivatives of
parameterized power transformations.
"""
abstract type DPowerTransform{R<:AbstractFloat} end

"""
TODO
"""
struct DBoxCox{R} <: DPowerTransform{R}
    λ::Uniform{R}
end

function (dg::DBoxCox)(λ, x)
    return x^(λ - one(x))
end

"""
TODO
"""
struct DYeoJohnson{R} <: PowerTransform{R}
    λ::Uniform{R}
end

function (dg::DYeoJohnson)(λ, x)
    # TODO
    return one(x)
end


end # module
