
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
struct SquaredExponential{R<:Real} <: Kernel
    γ::R
end

function (k::SquaredExponential)(τ)
    return exp(-(τ / k.γ)^2)
end
