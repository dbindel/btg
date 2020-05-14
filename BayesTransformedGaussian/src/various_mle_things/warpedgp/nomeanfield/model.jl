using LinearAlgebra
using Random, Distributions, Distances, StatsFuns
using Zygote

include("./kernel.jl")
include("./transform.jl")

struct Model{K<:AbstractKernel, G<:AbstractTransform}
    X::Matrix{Float64}
    Y::Vector{Float64}
    k::K
    g::G
    dims::Int
end


function make_fg(f, unpack)
    return function(F, G, x)
        v, back = pullback(x) do x
            f(unpack(x)...)
        end
        if G !== nothing
            G .= back(1.0)[1]
        end
        if F !== nothing
            return v
        end
        return nothing
    end
end

function logprob(mdl, ℓ, θ, ϵ, λ)
    n = length(mdl.Y)
    L = mdl.X / ℓ
    D = pairwise(SqEuclidean(), L; dims=mdl.dims)
    K = (τ -> radial(mdl.k, θ, τ)).(D) + max(ϵ, 1e-8) * I
    cholK = cholesky(Symmetric(K))
    Z = (y -> transform(mdl.g, λ, y)).(mdl.Y)
    lnJ = logjacobian(mdl.g, λ, mdl.Y)
    lnDet = logdet(cholK)
    return lnJ - lnDet / 2 - dot(Z, cholK \ Z) / 2 - n * log2π / 2
end
