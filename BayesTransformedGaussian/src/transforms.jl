import Base: inv

"""
Family of nonlinear transformation parametrized by λ that defaults to Box Cox
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

partial_x(::BoxCox, λ, x) = x ^ (λ - 1)
"""
Derivative of Box-Cox power transformation w.r.t x
"""
function boxCoxPrime(x, lambda=1)
    lambda==0 ? float(x).^(-1) : float(x).^(lambda .-1)
end

partial_xx(::BoxCox, λ, x) = (λ - 1) * x ^ (λ - 2)

"""
Second derivative of Box-Cox power transformation w.r.t x
"""
function boxCoxPrime2(x, lambda=1)
    lambda==0 ? -float(x).^(-2) : (lambda-1)*float(x).^(lambda-2)
end

"""
Derivative of Box-Cox power transformation w.r.t lambda
"""
partial_λ(::BoxCox, λ, x) = λ == 0 ? 0 : (λ * x ^ λ - x ^ λ + 1) / λ ^ 2
function boxCoxPrime_lambda(x, lambda=1)
    lambda==0 ? 0 : (lambda * float(x).^lambda .* log.(x) .- float(x).^lambda .+ 1)/lambda^2
end

"""
Second derivative of Box-Cox power transformation w.r.t lambda
"""
function partial_λλ(::BoxCox, λ, x)
    if λ == 0
        return 0
    end
    num = λ ^ 2 * x ^ λ * log(x) ^ 2 + 2 * x ^ λ - 2 * λ * x ^ λ * log(x) - 2
    return num / λ ^ 3
end
function boxCoxPrime_lambda2(x, lambda=1)
    lambda==0 ? 0 : float(x).^lambda .* (Base.log.(x)^2)/lambda - 
    (2*x.^lambda .* Base.log.(x))/lambda^2 + 2*(float(x).^lambda-1)/lambda^3
end

"""
Mixed derivative of Box-Cox power transformation w.r.t lambda and x
"""
partial_xλ(::Boxcox, λ, x) = x ^ (λ - 1) * log(x)
function boxCoxMixed_lambda_z(x, lambda=1)
    lambda==0 ? 0 : float(x).^(lambda-1) .* Base.log.(x) 
end

"""
Inverse Box Cox power transformation
"""
inv(::BoxCox, λ, y) = λ == 0 ? exp(y) : exp(log(λ * y + 1) / λ)
function invBoxCox(y, lambda=1)
    lambda==0 ? Base.exp.(y) : Base.exp.(Base.log.(lambda.*y.+1)./lambda)
end

#define BoxCox Object
boxCoxObj = nonlinearTransform(boxCox, boxCoxPrime, boxCoxPrime2, boxCoxPrime_lambda, boxCoxPrime_lambda2, boxCoxMixed_lambda_z, invBoxCox)

@doc raw"""
    TODO Unimplimented
"""
struct YeoJohnson end

@doc raw"""
    TODO Unimplimented
"""
struct ArandaOrdaz end
