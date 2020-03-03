
const Gaussian = Kernel.Lengthscale{T, Kernel.ExponentiatedQuadratic{T}} where T
Gaussian(σ::Real) = Kernel.Lengthscale(Kernel.ExponentiatedQuadratic(), σ)

@doc raw"""
    BoxCox(λ::Real)

The Box-Cox power transformation function with parameter λ.

```math
g_\lambda(x) = \begin{cases} 
    \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0\\ 
    \ln(x) & \lambda = 0
\end{cases}
```

External links
* [Power Transform on Wikipedia](https://en.wikipedia.org/wiki/Power_transform)

"""
struct BoxCox{T} <: Function
    λ::T
end

function (g::BoxCox)(x)
    return g.λ == 0 ? log(x) : expm1(log(x) * g.λ) / g.λ
end

function prime(g::BoxCox, x)
    return x^(g.λ - 1)
end

@doc raw"""
    TODO Unimplimented
"""
struct YeoJohnson{T} <: Function end

@doc raw"""
    TODO Unimplimented
"""
struct ArandaOrdaz{T} end
