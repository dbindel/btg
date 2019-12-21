"""
    BTGFuncs
BTGFuncs contains definitions for various functions and classes of functions used by
the BTG model    
"""
module BTGFuncs

using Distributions, Distances

export
    Correlation,
    IsotropicCorrelation,
    SquaredExponential,
    Transform,
    BoxCox,
    kernelmatrix,
    getparam,
    sampleparam,
    prime


"""
    Correlation 

Abstract supertype for all correlation functions
"""
abstract type Correlation end

"""
    IsotropicCorrelation <: Correlation

Abstract supertype for all isotropic correlation functions
"""
abstract type IsotropicCorrelation <: Correlation end

"""
    kernelmatrix(k::IsotropicCorrelation, θ, A, B)

Construct a correlation matrix by applying `k` with 
parameters `θ` pairwise to the rows of `A` and `B` 
"""
function kernelmatrix(k::IsotropicCorrelation, θ, A, B)
    return k.(θ, pairwise(Euclidean(), A, B, dims=1))
end

"""
    SquaredExponential(γ::Uniform)

The squared exponential kernel (aka the Gaussian kernel) with a length scale 
parameter `γ` given by a continuous uniform distribution.
    
    SquaredExponential(Uniform()) # γ is uniform over [0, 1]
    SquaredExponential(Uniform(a, b)) # γ is uniform over [a, b]

    TODO flesh out docs
"""
struct SquaredExponential <: IsotropicCorrelation
    γ::Uniform
end

function getparam(k::SquaredExponential, q)
    return (quantile(k.γ, q[1]),)
end

function sampleparam(k::SquaredExponential)
    return (rand(k.γ),)
end

function (k::SquaredExponential)(θ, τ)
    return exp(-(τ / θ[1])^2)
end

"""
    Transform 

Abstract supertype for all transformation functions
"""
abstract type Transform end

"""
    BoxCox(λ::Uniform)

The Box-Cox power transformation function with parameter λ given
by a continuous uniform distribution.
    
    BoxCox(Uniform()) # λ is uniform over [0, 1]
    BoxCox(Uniform(a, b)) # λ is uniform over [a, b]

    TODO flesh out docs
"""
struct BoxCox <: Transform
    λ::Uniform
end

function getparam(g::BoxCox, q)
    return (quantile(g.λ, q[1]),)
end

function sampleparam(g::BoxCox)
    return (rand(k.λ),)
end

function (g::BoxCox)(λ, x)
    return λ[1] == 0 ? log(x) : expm1(log(x) * λ) / λ
end

function prime(g::BoxCox, λ, x)
    return x^(λ[1] - 1)
end


end # module
