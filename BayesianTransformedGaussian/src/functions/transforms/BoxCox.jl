
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
struct BoxCox{R<:Real} <: Transform
    λ::R
end

function (g::BoxCox)(x)
    return g.λ == 0 ? log(x) : expm1(log(x) * g.λ) / g.λ
end

function prime(g::BoxCox, x)
    return x^(g.λ - 1)
end

function inverse(g::BoxCox, x)
    return g.λ == 0 ? exp(x) : exp(log(g.λ * y + 1) / g.λ)
end
