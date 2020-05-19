using LinearAlgebra
using StatsFuns
using Distances

function sumtanh(a, b, c, y)
    return y + dot(a, tanh.(b .* y .+ c))
end

function ∂sumtanh(a, b, c, y)
    return 1 + dot(a, b .* sech.(b .* y .+ c) .^ 2)
end

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

function logprob_gen(X, Y, k)
    n = length(Y)
    L = similar(X)
    K = similar(X, n, n)
    Z = similar(Y)
    C = similar(Y)
    dind = diagind(K)
    f = function (ℓ, ϵ, amp, a, b, c)
        copyto!(L, X)
        Z .= (y -> sumtanh(a, b, c, y)).(Y)
        cK = compute_cov!(L, K, nothing, ℓ, ϵ, amp, dind)
        lnd = logdet(cK)
        ldiv!(C, cK, Z)
        lnJ = sum(Y) do y ∂sumtanh(a, b, c, y) end
        return lnJ - lnd / 2 - dot(Z, C) / 2 - n * log2π / 2
    end
    ∂_K = similar(K)
    ∂ℓ = similar(X, size(X, 2))
    ∂a = similar(X, k)
    ∂b = similar(X, k)
    ∂c = similar(X, k)
    ∂_Z = similar(C)
    fg = function (ℓ, ϵ, amp, a, b, c)
        copyto!(L, X)
        Z .= (y -> sumtanh(a, b, c, y)).(Y)
        cK = compute_cov!(L, K, ∂_K, ℓ, ϵ, amp, dind)
        lnd = logdet(cK)
        ldiv!(C, cK, Z)
        ∂amp = ∂ofK!(∂_K, C, cK)
        for i in 1:length(ℓ)
            pairwise!(∂_K, SqEuclidean(), view(X, :, i:i); dims=1)
            ∂ℓ[i] = ∂ofK!(∂_K, C, cK)
        end
        for i in 1:k
            ∂_Z .= tanh.(b[i] .+ Y .+ c[i])
            ldiv!(cK, ∂_Z)
            ∂a[i] = -dot(Z, ∂_Z)
            ∂a[i] += sum(Y) do y b[i] * sech(b[i] * y + c[i]) ^ 2 end

            ∂_Z .= a[i] .* Y .* sech.(b[i] .+ Y .+ c[i]) .^ 2
            ldiv!(cK, ∂_Z)
            ∂b[i] = -dot(Z, ∂_Z)
            ∂b[i] += sum(Y) do y 
                ti = b[i] * y + c[i]
                a[i] * sech(ti) ^ 2 - 2 * a[i] * b[i] * tanh(ti) * sech(ti) ^ 2
            end

            ∂_Z .= a[i] .* sech.(b[i] .+ Y .+ c[i]) .^ 2
            ldiv!(cK, ∂_Z)
            ∂c[i] = -dot(Z, ∂_Z)
            ∂c[i] += sum(Y) do y
                ti = b[i] * y + c[i]
                2 * a[i] * b[i] * tanh(ti) * sech(ti) ^ 2
            end
        end
        copyto!(∂_K, I)
        ∂ϵ = ∂ofK!(∂_K, C, cK)
        lnJ = sum(Y) do y ∂sumtanh(a, b, c, y) end
        ret = lnJ  - lnd / 2 - dot(Z, C) / 2 - n * log2π / 2
        return ret, ∂ℓ, ∂ϵ, ∂amp, ∂a, ∂b, ∂c
    end
    return f, fg
end