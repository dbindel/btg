struct LengthScale{T}
    U::Matrix{T}
    X::Matrix{T}
    Du::Matrix{T}
    Dux::Matrix{T}
    Dx::Matrix{T}
end

function display(l::LengthScale)
    println("    U: Rescaled U")
    display(l.U)
    println("\n    X: Rescaled X")
    display(l.X)
    println("\n    Du: Distance between U")
    display(l.Du)
    println("\n    Dux: Distance between U and X")
    display(l.Dux)
    println("\n    Dx: Distance between X")
    display(l.Dx)
    return nothing
end
function lengthscale(dist, p, M, X)
    U = M \ X[:, 1:p]
    X′ = M \ X[:, p+1:end]
    Du = pairwise(dist, U; dims=2)
    Dux = pairwise(dist, U, X′; dims=2)
    Dx = pairwise(dist, X′; dims=2)
    return LengthScale(U, X′, Du, Dux, Dx)
end

function lengthscale(d, p, m)
    U = Matrix{T}(undef, d, p)
    X = Matrix{T}(undef, d, m)
    Du = Matrix{T}(undef, p, p)
    Dux = Matrix{T}(undef, p, m)
    Dx = Matrix{T}(undef, m, m)
    return LengthScale(U, X, Du, Dux, Dx)
end
function set!(l::LengthScale, dist, M, X)
    p = size(l.Du, 1)
    @views ldiv!(l.U, M, X[:, 1:p])
    @views ldiv!(l.X, M, X[:, p+1:end])
    pairwise!(l.Du, dist, l.U; dims=2)
    pairwise!(l.Dux, dist, l.U, l.X; dims=2)
    pairwise!(l.Dx, dist, l.X; dims=2)
    return l
end

abstract type AbstractKernel end

radial(k::AbstractKernel, θ::Tuple, τ) = radial(k, θ..., τ)

struct KernelMatrix{T}
    Ku::Matrix{T}
    Kux::Matrix{T}
    Kx::Matrix{T}
end

function display(km::KernelMatrix)
    println("    Ku: Kernel Evaluations between U")
    display(km.Ku)
    println("\n    Kux: Kernel Evaluations between U and X")
    display(km.Kux)
    println("\n    Kx: Kernel Evaluations between X")
    display(km.Kx)
    return nothing
end

function kernelmatrix(k::AbstractKernel, θ, l)
    Ku = (τ -> radial(k, θ, τ)).(l.Du)
    Kux = (τ -> radial(k, θ, τ)).(l.Dux)
    Kx = (τ -> radial(k, θ, τ)).(l.Dx)
    return KernelMatrix(Ku, Kux, Kx)
end
function kernelmatrix(T, p, m)
    Ku = Matrix{T}(undef, p, p)
    Kux = Matrix{T}(undef, p, m)
    Kx = Matrix{T}(undef, m, m)
    return KernelMatrix(Ku, Kux, Kx)
end
function set!(km::KernelMatrix, k, θ, l)
    km.Ku .= (τ -> radial(k, θ, τ)).(l.Du)
    km.Kux .= (τ -> radial(k, θ, τ)).(l.Dux)
    km.Kx .= (τ -> radial(k, θ, τ)).(l.Dx)
    return km
end

struct Gaussian <: AbstractKernel end

distance(::Gaussian) = SqEuclidean()
radial(::Gaussian, α::Number, τ) = α * exp(-τ / 2)

struct Spherical <: AbstractKernel end

distance(::Spherical) = SqEuclidean()
function radial(::Spherical, α::Number, r::Number, τ)
    τ′ = τ / r
    return τ′ > 1 ? zero(τ) : α * (1 - (3 / 2) * τ′ + (1 / 2) * τ′ ^ 3)
end
