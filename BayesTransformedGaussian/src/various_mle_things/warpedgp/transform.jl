abstract type AbstractTransform end

struct Output{T}
    z::Vector{T}
    rz::Vector{T}
end

function display(o::Output)
    println("    z: Transformed Observations")
    display(o.z)
    println("\n    rz: Reduced Transformed Observations")
    display(o.rz)
    return nothing
end

function output(W, g, λ, y, p)
    z = (x -> transform(g, λ, x)).(y)
    @views rz = z[p+1:end] .- W' * z[1:p]
    return Output(z, rz)
end
function output(T, n, m)
    z = Matrix{T}(undef, n)
    rz = Matrix{T}(undef, m)
    return Output(z, rz)
end
function set!(o::Output, W, g, λ, y)
    p = length(o.z) - length(o.rz)
    o.z .= (x -> radial(g, λ, x)).(y)
    @views o.rz .= o.z[:, p+1:end] .- W * o.z[:, 1:p]
    return o
end

logjacobian(g::AbstractTransform, λ::Tuple, y) = sum(x -> log∂(g, λ, x), y)
transform(g::AbstractTransform, λ::Tuple, y) = transform(g, λ..., y)
log∂(g::AbstractTransform, λ::Tuple, y) = log∂(g, λ..., y)

struct Identity <: AbstractTransform end

transform(::Identity, y) = y
log∂(::Identity, _) = 0

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
