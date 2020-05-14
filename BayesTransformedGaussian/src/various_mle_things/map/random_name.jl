using Distributions
using Distances
using Random
using LinearAlgebra
using Zygote
using Zygote: @adjoint

include("./kernel.jl")
include("./transform.jl")
include("./covariate.jl")

abstract type AbstractParameter{T} end

struct Parameter{T}
    param::Vector{T}
    probs::Vector{Float64}
end

mutable struct ScalarParameter{T<:Real} <: AbstractParameter{T}
    param::T
    dist::Distribution{Univariate,Continuous}
end

struct VectorParameter{T<:Real} <: AbstractParameter{T}
    param::Vector{T}
    dist::Vector{Distribution{Univariate,Continuous}}
end

log_jacobian(g, λ, y) = sum(x -> log∂(g, λ, x), y)

function logprob(k::AbstractKernel, g::AbstractTransform, f::AbstractCovariate, U::AbstractMatrix, X::AbstractMatrix, yu::AbstractVector, yx::AbstractVector)
    Fu::Matrix{Float64} = f(U)
    W::Matrix{Float64} = Fu \ f(X)
    h = let W = W, k = k, U = U, X = X, g = g, yu = yu, yx = yx, Fu = Fu
        (θ, λ) -> logprob(k, θ, g, λ, U, Fu, X, W, yu, yx)
    end
    return h
end

Zygote.@adjoint function mul!(
    C::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    α::Real,
    β::Real,
)
    tmp, back = pullback(C, A, B, α, β) do C, A, B, α, β
        (A * B) .* α .+ C .* β 
    end
    return C, c::AbstractMatrix -> back(c)
end

function update_Kxx′!(
    Kxx′::AbstractMatrix,
    B::AbstractMatrix,
    Kux::AbstractMatrix,
    W::AbstractMatrix,
    W′::AbstractMatrix
)
    mul!(Kxx′, W', B, -1, 1)
    mul!(Kxx′, Kux', W′, -1, 1)
    return Kxx′
end

function update_Kx′!(
    Kx′::AbstractMatrix,
    B::AbstractMatrix,
    Kux′::AbstractMatrix,
    W′::AbstractMatrix
) # B = Kux′ .- Ku * W′
    mul!(Kx′, (W′)', B, -1, 1)
    mul!(Kx′, (Kux′), W′, -1, 1)
    return Kx′
end

struct Block
    Ku::Matrix{Float64}
    Kux::Matrix{Float64}
    rKx::Matrix{Float64}
end

function logprob(k, θ, g, λ, U, Fu, X, W, yu, yx)
    p, m = size(W)
    Ku = correlation(k, θ, U)
    Kux = correlation(k, θ, U, X)
    Kx = correlation(k, θ, X)
    B = Kux .- Ku * W
    rKx = Kx .- W' * B .- Kux' * W
    zu = (x -> g(λ, x)).(yu)
    zx = (x -> g(λ, x)).(yx)
    z = zx .- W' * zu
    c = rKx \ z
    lnJλ = log_jacobian(g, λ, yu) + log_jacobian(g, λ, yx)
    lnDet = 2 * logabsdet(Fu)[1] + logdet(rKx)
    lnQ = log(dot(z, c))
    return m / (m + p) * lnJλ - m / 2 * lnQ - 0.5 * lnDet
end
