using LinearAlgebra, Arpack, IterativeSolvers
using Random, Distributions, Distances
using Zygote
using Zygote: @adjoint
import Base: display, size
import LinearAlgebra: logabsdet, logdet, eltype, mul!

include("./kernel.jl")
include("./transform.jl")
include("./covariate.jl")
include("./opt.jl")

struct Data{T}
    X::Matrix{T} # d x n
    Fu::LU{Float64, Matrix{T}} # p x p
    W::Matrix{T} # p x n - p
    y::Vector{T} # n
end

function display(d::Data)
    println("    X: Observation Locations")
    display(d.X)
    println("\n    Fu: Covariate Basis")
    display(d.Fu)
    println("\n    W: Transformed Covariates")
    display(d.W)
    println("\n    y: Observations")
    display(d.y)
    return nothing
end
    

function data(f, X, y)
    Fx = covariate(f, X)
    p, n = size(Fx)
    Fu = lu(Fx[:, 1:p])
    W = Fu \ Fx[:, p+1:end]
    return Data(X, Fu, W, y)
end

mutable struct KernelSystem{
    T, K<:AbstractKernel, G<:AbstractTransform, F<:AbstractCovariate
}
    f::F
    k::K
    g::G
    d::Data{T}
    l::LengthScale{T}
    km::KernelMatrix{T}
    ϵ::T
    o::Output{T}
end

function display(ks::KernelSystem)
    println("        f: Covariate Function")
    display(ks.f)
    println("\n        k: Kernel")
    display(ks.k)
    println("\n        g: Transformation")
    display(ks.g)
    println("\n        d: Data")
    display(ks.d)
    println("\n        l: Scaled Distances")
    display(ks.l)
    println("\n        km: Kernel Matrix")
    display(ks.km)
    println("\n        ϵ: Nugget term")
    display(ks.ϵ)
    println("\n        o: Transformed Observations")
    display(ks.o)
    return nothing
end
    

function kernelsystem(f, k, g, d, ℓ, θ, ϵ, λ)
    p, m = size(d.W)
    l = lengthscale(distance(k), p, ℓ, d.X)
    km = kernelmatrix(k, θ, l)
    o = output(d.W, g, λ, d.y, p)
    return KernelSystem(f, k, g, d, l, km, ϵ, o)
end
schur(ks::KernelSystem) = Schur(ks.d.W, ks.km.Ku, ks.km.Kux, ks.km.Kx, ks.ϵ)
function logabsdet(ks::KernelSystem)
    s = (-1.0) ^ (size(ks.d.Fu, 1) % 2)
    return 2 * logabsdet(ks.d.Fu)[1] * logdet(schur(ks))
end
function logabsdet(ks::KernelSystem, rKx)
    s = (-1.0) ^ (size(ks.d.Fu, 1) % 2)
    return 2 * logabsdet(ks.d.Fu)[1] * logdet(rKx)
end

struct Schur{T}
    W::Matrix{T}
    Ku::Matrix{T}
    Kux::Matrix{T}
    Kx::Matrix{T}
    ϵ::T
end

function Matrix(sc::Schur)
    B = sc.Kux .- sc.Ku * sc.W .- sc.ϵ .* sc.W
    return sc.Kx .- sc.W' * B .- sc.Kux' * sc.W + sc.ϵ * I
end
eltype(::Schur{T}) where T = T
size(sc::Schur, args...) = size(sc.Kx, args...)
function mul!(y, sc::Schur, x, α, β)
    # (Kx - Kux'W - W'Kux + W'(Ku + ϵI)W + ϵI)x * α + y * β
    p, m = size(sc.W)
    v1 = similar(x, p)
    v2 = similar(x, p)

    y .= y .* β .+ x .* (α * sc.ϵ)

    mul!(y, sc.Kx, x, α, 1)

    mul!(v1, sc.W, x)
    mul!(v2, v1, ϵ)
    mul!(v2, sc.Ku, v1, 1, 1)

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

function lanczos_logdet(A, k; maxiter, tol, nsamples)
    z = Vector{eltype(A)}(undef, size(A, 1))
    return _lanczos_logdet!(z, 0, A, k;
                            maxiter=maxiter, tol=tol, nsamples=nsamples)
end

logdet(sc::Schur) = lanczos_logdet(sc, max(20, size(sc, 1));
                                   maxiter=1000, tol=0.0, nsamples=50)

function logprob(ks, λ, logprior)
    p, m = size(ks.d.W)
    lnJ = logjacobian(ks.g, λ, ks.o.z)
    rKx = cholesky(Symmetric(Matrix(schur(ks))))
    lnq = dot(ks.o.rz, rKx \ ks.o.rz)
    lnDet = logabsdet(ks, rKx)[1]
    return m / (m + p) * lnJ - m / 2 * lnq - lnDet / 2 + logprior
end
