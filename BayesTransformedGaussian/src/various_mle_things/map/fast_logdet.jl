using LinearAlgebra, Arpack, IterativeSolvers
using Random, Distributions

struct Schur{
    T,
    M<:AbstractMatrix{T},
    N<:AbstractMatrix{T},
    O<:AbstractMatrix{T},
    P<:AbstractMatrix{T}
}
    W::M
    Ku::N
    Kux::O
    Kx::P
    ϵ::T
end

function Matrix(sc::Schur)
    return sc.Kx .- sc.W' * (sc.Kux .- sc.Ku * sc.W) .- sc.Kux' * sc.W + sc.ϵ * I
end
eltype(::Schur{T}) where T = T
size(sc::Schur, args...) = size(sc.Kx, args...)
function mul!(y, sc::Schur, x, α, β)
    p, n = size(sc.W)
    v1 = similar(x, p)
    v2 = similar(x, p)

    y .= y .* β .+ x .* (α * sc.ϵ)

    mul!(y, sc.Kx, x, α, 1)
    
    mul!(v1, sc.W, x)
    mul!(v2, sc.Ku, v1)
    
    mul!(y, sc.W', v2, α, 1)
    mul!(y, sc.Kux', v1, -α, 1)

    mul!(v1, sc.Kux, x)

    mul!(y, sc.W', v1, -α, 1)

    return y
end

function lanczos_arpack(A, k, v; maxiter, tol)
    T = eltype(A)
    n = size(A, 1)
    mulA! = (y, x) -> mul!(y, A, x) 
    id = x -> x
    # in: (T, mulA!, mulB, solveSI, n, issym, iscmplx, bmat,
    #            nev, ncv, whichstr, tol, maxiter, mode, v0)
    # out: (resid, v, ldv, iparam, ipntr, workd, workl, lworkl, rwork, TOL)
    out = Arpack.aupd_wrapper(T, mulA!, id, id, n, true, false, "I",
                       1, k, "LM", tol, maxiter, 1, v)

    α = out[7][k+1:2*k-1]
    β = out[7][2:k-1]
    
    return out[2], α, β, out[1]
end

function _lanczos_logdet!(z, acc, sc, k; maxiter, tol, nsamples)
    for i in 1:nsamples
        rand!(Normal(), z)
        z .= sign.(z)
        Q, α, β, resid = lanczos_arpack(sc, k, z; maxiter=maxiter, tol=tol)
        T = SymTridiagonal(α, β)
        Λ = eigen(T)
        wts = Λ.vectors[1, :] .^ 2 .* norm(z) ^ 2
        acc += dot(wts, log.(Λ.values))
    end
    return acc / nsamples
end

function lanczos_logdet(sc, k; maxiter, tol, nsamples)
    z = Vector{eltype(sc)}(undef, size(sc, 1))
    return _lanczos_logdet!(z, 0, sc, k;
                            maxiter=maxiter, tol=tol, nsamples=nsamples)
end
