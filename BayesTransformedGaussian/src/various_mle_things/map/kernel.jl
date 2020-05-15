abstract type AbstractKernel end

function covariance!(b, k, θ, X, Y)
    distance!(b, k, θ, X, Y)
    b .= (x -> radial(k, θ, x)).(b)
    return b
end

Zygote.@adjoint function covariance!(b, k, θ, X, Y)
    tmp, back = pullback(b, k, θ, X, Y) do b, k, θ, X, Y
        d = distance!(b, k, θ, X, Y)
        (x -> radial(k, θ, x)).(d)
    end
    b .= tmp
    return b, c -> back(c)
end

function jitter!(b, ϵ; rescale=true)
    for i in diagind(b)
        b[i] += ϵ
    end
    if rescale
        b ./= b[1, 1]
    end
    return b
end

Zygote.@adjoint function jitter!(b, ϵ; rescale=true)
    tmp, back = pullback(b, ϵ) do b, ϵ
        c = b + UniformScaling(ϵ)
        if rescale
            c ./ c[1]
        else
            c
        end
    end
    b .= tmp
    return b, c -> back(c)
end

sqeuclidean!(b, X, Y, ℓ) = sqeuclidean!(b, X ./ ℓ, Y ./ ℓ)
function sqeuclidean!(b, X, Y)
    mul!(b, X', Y, -2, 0)
    r1 = sum!(abs2, similar(X, 1, size(X, 2)), X)
    r2 = sum!(abs2, similar(Y, 1, size(Y, 2)), Y)
    b .+= r1' .+ r2
    return b
end

Zygote.@adjoint function sqeuclidean!(b, X, Y)
    tmp, back = pullback(X, Y) do A, B
        pairwise(SqEuclidean(), A, B; dims=2)
    end
    b .= tmp
    return b, c::AbstractMatrix -> (nothing, back(c)...)
end

struct Gaussian <: AbstractKernel end
distance!(b, ::Gaussian, θ, X, Y) = sqeuclidean!(b, X, Y, θ[1])
radial(::Gaussian, _, τ) = exp(-τ / 2)
