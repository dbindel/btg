"""
    BTGFuncs

The `BTGFuncs` module provides transformation, covariate, and correlation 
functions used in BTG models.
"""
module BTGFuncs

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
    θ1::R
    θ2::R
end

function (k::ExpCorr)(τ)
    return k.θ1^(τ^k.θ2)
end

"""
TODO
"""
struct MaternCorr{R} <: IsotropicCorrelation{R}
    θ1::R
    θ2::R
end

function (k::MaternCorr)(τ) 
    # TODO
    return one(τ)
end

"""
TODO
"""
struct RationalCorr{R} <: IsotropicCorrelation{R}
    θ1::R
    θ2::R
end

function (k::RationalCorr)(τ)
    return if τ == zero(τ)
        one(τ)
    else
        θ1 = -log(k.θ1)
        θ2 = -log(k.θ2)
        τ′ = τ / θ1
        (one(τ′) + τ′^2)^(-θ2)
    end
end

"""
TODO
"""
struct SphericalCorr{R} <: IsotropicCorrelation{R}
    θ::R
end

function (k::SphericalCorr)(τ)
    return if τ == zero(τ)
        one(τ)
    else
        θ = -log(k.θ)
        if τ <= θ
            τ′ = τ / θ
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
    λ::R
end

function (g::BoxCox)(x)
    return g.λ == zero(g.λ) ? log(x) : expm1(log(x) * λ) / λ
end

"""
TODO
"""
struct YeoJohnson{R} <: PowerTransform{R}
    λ::R
end

function (g::YeoJohnson)(x)
    return if g.λ != zero(g.λ) && x >= zero(x)
        expm1(g.λ * log1p(x)) / g.λ
    elseif g.λ == zero(g.λ) && x >= zero(x)
        log1p(x)
    elseif g.λ != oftype(g.λ, 2) && x < zero(x)
        -expm1((2 - g.λ) * log1p(-x)) / (2 - g.λ)
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
    λ::R
end

function (dg::DBoxCox)(x)
    return x^(dg.λ - one(x))
end

"""
TODO
"""
struct DYeoJohnson{R} <: PowerTransform{R}
    λ::R
end

function (dg::DYeoJohnson)(x)
    # TODO
    return one(x)
end


end # module
