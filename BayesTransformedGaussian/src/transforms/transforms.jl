import Base: inv

"""
Family of nonlinear transformation parametrized by λ that defaults to Box Cox.
N.B. We assume in derivatives.jl that the nonlinear transformation f is monotonic increasing. 
"""



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

...
# Arguments
* `x::float64`: evaluation point
* `lambda::float=1.0`: lambda hyperparameter

"""
abstract type NonlinearTransform end

struct BoxCox <:NonlinearTransform end 

struct anotherone<:NonlinearTransform end

(::BoxCox)(x, λ::Real) = λ == 0 ? log.(x) : expm1.(log.(x) .* λ) ./ λ

partialx(::BoxCox, x, λ) =  λ==0 ? float(x).^(-1) : float(x).^(λ .-1)
partialxx(::BoxCox, x, λ) = λ==0 ? -float(x).^(-2) : (λ-1)*float(x).^(λ-2)
partialλ(::BoxCox, x, λ) = λ==0 ? 0 : (λ * float(x).^λ .* log.(x) .- float(x).^λ .+ 1)/λ^2
partialλλ(::BoxCox, x, λ) = (λ==0 ? 0 : float(x).^λ .* (Base.log.(x)^2)/λ - 
                            (2*x.^λ .* Base.log.(x))/λ^2 + 2*(float(x).^λ-1)/λ^3)
partialxλ(::BoxCox, x, λ) = λ==0 ? 0 : float(x).^(λ-1) .* Base.log.(x)
inverse(::BoxCox, x, λ) = λ==0 ? Base.exp.(x) : Base.exp.(Base.log.(λ.*x.+1)./λ)
