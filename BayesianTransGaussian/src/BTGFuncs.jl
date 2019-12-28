"""
    BTGFuncs
BTGFuncs contains definitions for various functions and classes of functions used by
the BTG model    
"""
module BTGFuncs

using Distributions, Distances

export
    Parametric,
    Correlation,
    IsotropicCorrelation,
    SquaredExponential,
    Transform,
    BoxCox,
    Covariate,
    Identity,
    kernelmatrix,
    prime

"""
    Parametric

Abstract supertype for all parametric functions
"""
abstract type Parametric end

"""
    Correlation 

Abstract supertype for all correlation functions
"""
abstract type Correlation <: Parametric end

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

The squared exponential kernel (aka the Gaussian kernel or the radial basis
function kernel) with a length scale 
parameter `γ` given by a continuous uniform distribution.

```math
k_\\theta(\\tau) = \\exp\\Big\\lbrace -\\Big(\\frac{\\tau}{\\theta}\\Big)^2\\Big\\rbrace
```

```julia    
SquaredExponential(Uniform()) # γ is uniform over [0, 1]
SquaredExponential(Uniform(a, b)) # γ is uniform over [a, b]

sampleparam(k) # A random sample of the parameters of k
k(θ, τ) # The kernel evaluated on the distance τ with parameters θ
```

External links
* [RBF Kernel on Wikipedia](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)

"""
struct SquaredExponential <: IsotropicCorrelation
    γ::Union{Uniform, Exponential, Gamma}
end

function (k::SquaredExponential)(θ, τ)
    return exp(-(τ / θ[1])^2)
end

"""
    Transform 

Abstract supertype for all transformation functions
"""
abstract type Transform <: Parametric end

"""
    BoxCox(λ::Uniform)

The Box-Cox power transformation function with parameter λ given
by a continuous uniform distribution.

```math
g_\\lambda(x) = \\begin{cases} 
    \\frac{x^\\lambda - 1}{\\lambda} & \\lambda \\neq 0\\\\ 
    \\ln(x) & \\lambda = 0
\\end{cases}
```

```julia    
BoxCox(Uniform()) # λ is uniform over [0, 1]
BoxCox(Uniform(a, b)) # λ is uniform over [a, b]

sampleparam(g) # A random sample of the parameters of g
g(λ, x) # The transformation evaluated on x with parameters y
prime(g, λ, x) # The derivative of the transformation evaluated on x with parameters y
```

External links
* [Power Transform on Wikipedia](https://en.wikipedia.org/wiki/Power_transform)

"""
struct BoxCox <: Transform
    λ::Union{Uniform, Normal}
end

function (g::BoxCox)(λ, x)
    return λ[1] == 0 ? log(x) : expm1(log(x) * λ[1]) / λ[1]
end

function prime(g::BoxCox, λ, x)
    return x^(λ[1] - 1)
end

"""
    Covariate

Abstract supertype for all covariate functions
"""
abstract type Covariate end

"""
    Identity

A covariate function which returns its input as-is
"""
struct Identity <: Covariate end

function (f::Identity)(x)
    return x
end

end # module
