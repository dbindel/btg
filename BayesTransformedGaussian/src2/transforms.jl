import Base: inv

@doc raw"""
"""
abstract type Transform end

@doc raw"""
"""
struct BoxCox <: Transform end
(::BoxCox)(λ, y) = λ == 0 ? log(y) : expm1(log(y) * λ) / λ
inv(::BoxCox, λ, z) = λ == 0 ? exp(z) : exp(log(λ * z + 1) / λ)
