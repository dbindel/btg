using LinearAlgebra
using StatsFuns
using Distances

function ∂ofK!(∂K, C, cK)
    ret = - dot(∂K * C, C) / 2
    ldiv!(cK, ∂K)
    ret -= tr(∂K) / 2
    return ret
end

function compute_cov!(X, K, ∂K, ℓ, ϵ, amp, dind)
    copyto!(X, X)
    rdiv!(X, Diagonal(ℓ))
    pairwise!(K, SqEuclidean(), X; dims=1)
    K .= exp.(.-K)
    if !isnothing(∂K)
        copyto!(∂K, K)
    end
    K .= amp .* K
    for i in dind
        K[i] += ϵ
    end
    cK = cholesky!(Symmetric(K))
end

function logprob_gen(X, Y)
    n = length(Y)
    L = similar(X)
    K = similar(X, n, n)
    C = similar(Y)
    dind = diagind(K)
    f = function (ℓ, ϵ, amp)
        copyto!(L, X)
        cK = compute_cov!(L, K, nothing, ℓ, ϵ, amp, dind)
        lnd = logdet(cK)
        ldiv!(C, cK, Y)
        return - lnd / 2 - dot(Y, C) / 2 - n * log2π / 2
    end
    ∂_K = similar(K)
    ∂ℓ = similar(X, size(X, 2))
    fg = function (ℓ, ϵ, amp)
        copyto!(L, X)
        cK = compute_cov!(L, K, ∂_K, ℓ, ϵ, amp, dind)
        lnd = logdet(cK)
        ldiv!(C, cK, Y)
        ∂amp = ∂ofK!(∂_K, C, cK)
        for i in 1:length(ℓ)
            pairwise!(∂_K, SqEuclidean(), view(X, :, i:i); dims=1)
            ∂ℓ[i] = ∂ofK!(∂_K, C, cK)
        end
        copyto!(∂_K, I)
        ∂ϵ = ∂ofK!(∂_K, C, cK)
        ret = - lnd / 2 - dot(Y, C) / 2 - n * log2π / 2
        return ret, ∂ℓ, ∂ϵ, ∂amp
    end
    return f, fg
end
