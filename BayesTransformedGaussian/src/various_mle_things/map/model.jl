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

logjacobian(g, λ, y) = sum(x -> log∂(g, λ, x), y)

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
    C .= tmp
    return C, c::AbstractMatrix -> back(c)
end

Zygote.@adjoint function ldiv!(A, b)
    tmp, back = pullback(A, b) do A, b
        A \ b
    end
    b .= tmp
    return b, c::AbstractArray -> back(c)
end
Zygote.@adjoint function ldiv!(c, A, b)
    tmp, back = pullback(A, b) do A, b
        A \ b
    end
    c .= tmp
    return b, c::AbstractArray -> (nothing, back(c)...)
end

Zygote.@adjoint function cholesky!(A)
    tmp, back = pullback(A) do A
        cholesky(A)
    end
    return tmp, c::AbstractArray -> back(c)
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
    mul!(Kx′, (Kux′)', W′, -1, 1)
    return Kx′
end

struct Border
    U::Matrix{Float64} # p observation locations
    Fu::LU{Float64, Matrix{Float64}} # p x p covariates at U
    X::Matrix{Float64} # n - p observation locations
    W::Matrix{Float64} # Fu \ Fx
    y::Vector{Float64} # length n vector
end
function border(f, U, X, y)
    Fu = lu!(f(U))
    W = ldiv!(Fu, f(X))
    return Border(U, Fu, X, W, y)
end

struct Block
    Ku::Matrix{Float64}
    Kux::Matrix{Float64}
    B::Matrix{Float64}
    rKx::Matrix{Float64}
end
function block(p, m)
    Ku = Matrix{Float64}(undef, p, p)
    Kux = Matrix{Float64}(undef, p, m)
    B = Matrix{Float64}(undef, p, m)
    rKx = Matrix{Float64}(undef, m, m)
    return Block(Ku, Kux, B, rKx)
end

struct RightHandSide
    z::Vector{Float64}
    rz::Vector{Float64}
    rc::Vector{Float64}
end
function righthandside(p, m)
    z = Vector{Float64}(undef, p + m)
    rz = Vector{Float64}(undef, m)
    rc = Vector{Float64}(undef, m)
    return RightHandSide(z, rz, rc)
end

function logabsdet(Fu, rKx)
    s = (-1.0) ^ (size(Fu, 1) % 2)
    return 2 * logabsdet(Fu)[1] + logdet(rKx), s
end

function reduced_system(k, θ, ϵ, br, bl)
    p, m = size(br.W)
    Ku = jitter!(covariance!(bl.Ku, k, θ, br.U, br.U), ϵ)
    Kux = covariance!(bl.Kux, k, θ, br.U, br.X)
    Kx = jitter!(covariance!(bl.rKx, k, θ, br.X, br.X), ϵ)
    B = mul!(copyto!(bl.B, Kux), bl.Ku, br.W, -1, 1)
    rKx = cholesky!(Symmetric(update_Kx′!(Kx, B, Kux, br.W)))
    return rKx
end

function solve_reduced(g, λ, rKx, br, rhs)
    p = size(br.U, 2)
    rhs.z .= (x -> transform(g, λ, x)).(br.y)
    @views rz = mul!(copyto!(rhs.rz, rhs.z[p+1:end]), br.W', rhs.z[1:p], -1, 1)
    return ldiv!(rKx, copyto!(rhs.rc, rz)), rz
end

function logprob(k, θ, ϵ, g, λ, br, bl, rhs)
    p, m = size(br.W)
    lnJ = logjacobian(g, λ, br.y)
    rKx = reduced_system(k, θ, ϵ, br, bl)
    rc, rz = solve_reduced(g, λ, rKx, br, rhs)
    lnQ = log(dot(rc, rz))
    lnDet = logabsdet(br.Fu, rKx)[1]

    return m / (m + p) * lnJ - m / 2 * lnQ - 1 / 2 * lnDet
end
