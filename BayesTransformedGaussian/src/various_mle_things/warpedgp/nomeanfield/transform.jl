abstract type AbstractTransform end

function logjacobian(g::AbstractTransform, λ::Tuple, y)
    return sum(x -> log∂(g, λ, x), y)
end

struct DoNothing <: AbstractTransform end

transform(::DoNothing, y) = y
log∂(::DoNothing, _) = 0

struct BoxCox <: AbstractTransform end

transform(::BoxCox, λ::Number, y) = λ == 0 ? log(y) : expm1(log(y) * λ) / λ
log∂(::BoxCox, λ::Number, y) = (λ - 1) * log(y)

struct YeoJohnson <: AbstractTransform end

function transform(::YeoJohnson, λ::Number, y)
    λ′ = y >= 0 ? λ : 2 - λ
    return sign(y) * (λ′ == 0 ? log1p(abs(y)) : expm1(log1p(abs(y)) * λ′) / λ′)
end
log∂(::YeoJohnson, λ::Number, y) = sign(y) * (λ - 1) *  log(abs(y))

struct SumTanh <: AbstractTransform end

function transform(::SumTanh, α::Vector, β::Vector, c::Vector, y)
    return sum(α .* tanh.(β .* (y .+ c)))
end
function log∂(::SumTanh, α::Vector, β::Vector, c::Vector, y)
    return sum(α .* β .* sech.(β .* (y .+ c)) .^ 2)
end
