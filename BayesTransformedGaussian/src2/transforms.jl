import Base: inv

@doc raw"""
"""
abstract type Transform end

@doc raw"""
"""
struct BoxCox <: Transform end

(::BoxCox)(λ, y) = λ == 0 ? log(y) : expm1(log(y) * λ) / λ
inv(::BoxCox, λ, z) = λ == 0 ? exp(z) : exp(log(λ * z + 1) / λ)

partial_x(::BoxCox, λ, x) = x ^ (λ - 1)
partial_xx(::BoxCox, λ, x) = (λ - 1) * x ^ (λ - 2)

partial_xλ(::BoxCox, λ, x) = x ^ (λ - 1) * log(x)

partial_λ(::BoxCox, λ, y) = λ == 0 ? 0 : (λ * x ^ λ - x ^ λ + 1) / λ ^ 2
function partial_λλ(::BoxCox, λ, x)
    if λ == 0
        return 0
    end
    num = λ ^ 2 * x ^ λ * log(x) ^ 2 + 2 * x ^ λ - 2 * λ * x ^ λ * log(x) - 2
    return num / λ ^ 3
end


@doc raw"""
    TODO Unimplimented
"""
struct YeoJohnson <: Transform end

@doc raw"""
    TODO Unimplimented
"""
struct ArandaOrdaz <: Transform end
