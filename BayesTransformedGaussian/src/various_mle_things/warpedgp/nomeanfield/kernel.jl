
abstract type AbstractKernel end

radial(k, θ::Tuple, τ) = radial(k, θ..., τ)

struct Gaussian <: AbstractKernel end

radial(::Gaussian, α::Number, τ) = α * exp(-τ / 2)

struct Spherical <: AbstractKernel end

function radial(::Spherical, α::Number, r::Number, τ)
    τ′ = τ / r
    return τ′ > 1 ? zero(τ) : α * (1 - (3 / 2) * τ′ + (1 / 2) * τ′ ^ 3)
end
