abstract type AbstractTransform end

struct BoxCox <: AbstractTransform end

transform(::BoxCox, λ, y) = λ[1] == 0 ? log(y) : expm1(log(y) * λ[1]) / λ[1]
log∂λ(::BoxCox, λ, y) = (λ[1] - 1) * log(y)

struct IdentityTransform <: AbstractTransform end

transform(::IdentityTransform, _, y) = y
log∂(::IdentityTransform, _, _) = 0

struct YeoJohnson <: AbstractTransform end

function transform(::YeoJohnson, λ, y)
    λ′ = y >= 0 ? λ[1] : 2 - λ[1]
    return sign(y) * (λ′ == 0 ? log1p(abs(y)) : expm1(log1p(abs(y)) * λ′) / λ′)
end
log∂(::YeoJohnson, λ, y) = ((y >= 0 ? λ[1] : 2 - λ[1]) - 1) *  log(abs(y))

