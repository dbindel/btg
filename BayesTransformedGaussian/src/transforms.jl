import Base: inv

"""
Family of nonlinear transformation parametrized by λ that defaults to Box Cox.
N.B. We assume in derivatives.jl that the nonlinear transformation f is monotonic increasing. 
"""
struct nonlinearTransform{T1<:Function, T2<:Function, T3<:Function, T4<:Function, T5<:Function, T6<:Function, T7<:Function}
    f::T1
    df::T2
    d2f::T3 
    df_hyp::T4 
    d2f_hyp::T5 
    d_mixed::T6 
    inv::T7 
end


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
struct BoxCox end

(::BoxCox)(λ, x) = λ == 0 ? log(x) : expm1(log(x) * λ) / λ

function boxCox(x, lambda=1)
    lambda == 0 ? Base.log.(x) : (float(x).^lambda.-1)./lambda
end

"""
Derivative of Box-Cox power transformation w.r.t x
"""
function boxCoxPrime(x, lambda=1)
    lambda==0 ? float(x).^(-1) : float(x).^(lambda .-1)
end

"""
Second derivative of Box-Cox power transformation w.r.t x
"""
function boxCoxPrime2(x, lambda=1)
    lambda==0 ? -float(x).^(-2) : (lambda-1)*float(x).^(lambda-2)
end

"""
Derivative of Box-Cox power transformation w.r.t lambda
"""
function boxCoxPrime_lambda(x, lambda=1)
    lambda==0 ? 0 : (lambda * float(x).^lambda .* log.(x) .- float(x).^lambda .+ 1)/lambda^2
end

"""
Second derivative of Box-Cox power transformation w.r.t lambda
"""
function boxCoxPrime_lambda2(x, lambda=1)
    lambda==0 ? 0 : float(x).^lambda .* (Base.log.(x)^2)/lambda - 
    (2*x.^lambda .* Base.log.(x))/lambda^2 + 2*(float(x).^lambda-1)/lambda^3
end

"""
Mixed derivative of Box-Cox power transformation w.r.t lambda and x
"""

function boxCoxMixed_lambda_z(x, lambda=1)
    lambda==0 ? 0 : float(x).^(lambda-1) .* Base.log.(x) #remove first term completely?
end

"""
Inverse Box Cox power transformation
"""
function invBoxCox(y, lambda=1)
    lambda==0 ? Base.exp.(y) : Base.exp.(Base.log.(lambda.*y.+1)./lambda)
end

#define BoxCox Object
boxCoxObj = nonlinearTransform(boxCox, boxCoxPrime, boxCoxPrime2, boxCoxPrime_lambda, boxCoxPrime_lambda2, boxCoxMixed_lambda_z, invBoxCox)

