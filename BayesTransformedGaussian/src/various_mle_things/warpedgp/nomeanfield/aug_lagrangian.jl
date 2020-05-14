using LinearAlgebra

include("./grad_projection.jl") 

mutable struct AugmentedLagrangian
    f::Function
    c::Function
    λ::Vector{Float64}
    μ::Float64
end

function (aug::AugmentedLagrangian)(x, ::Val{false})
    fx = aug.f(x, Val(false))
    cx = aug.c(x, Val(false))
    return fx - (dot(cx, aug.λ) - aug.μ * dot(cx, cx) / 2)
end

function (aug::AugmentedLagrangian)(x, ::Val{true})
    fx, gx = aug.f(x, Val(true))
    cx, Jx = aug.c(x, Val(true))

    ret = fx - (dot(cx, aug.λ) - aug.μ * dot(cx, cx) / 2)
    ret′ = mul!(gx, Jx', aug.λ, -1, 1)
    ret′ = mul!(ret′, Jx', cx, aug.μ, 1)

    return ret, ret′
end

function descent(f, x, rtol)
    fx, gx = f(x, Val(true))

    for i in 1:100
        if norm(gx) <= rtol
            return x, (x - (x + gx))
        end
        x .-= 0.01 .* gx
        fx, gx = f(x, Val(true))
    end
    return x, (x - (x + gx))
end

function _augmented_lagrangian!(
    aug, c, xk, ωk, ηk;
    lbound, ubound,
    maxiter, rtol, ctol, sr1tol, μpenalty, ηpenalty, ηupdate, monitor
)
    for k in 1:maxiter
        xk, r = descent(aug, xk, rtol)
        c_val = c(xk, Val(false))
        monitor(xk, norm(r))
        if norm(c_val) <= ηk
            if norm(c_val) <= ctol && norm(r) <= rtol
                return xk, true
            end
            aug.λ .-= aug.μ .* c_val
            ηk /= aug.μ ^ ηupdate
            ωk /= aug.μ
        else
            aug.μ *= μpenalty
            ηk = 1 / aug.μ ^ ηpenalty
            ωk = 1 / aug.μ
        end
    end
    return xk, false
end

function augmented_lagrangian(
    f, c, x0, λ0, μ0;
    lbound=fill(-Inf, size(x0)), ubound=fill(Inf, size(x0)),
    μpenalty=100, ηpenalty=0.1, ηupdate=0.9,
    ctol=1e-8, rtol=1e-8, sr1tol=1e-8,
    maxiter=100, monitor=(x, rnorm) -> nothing
)
    aug = AugmentedLagrangian(f, c, λ0, μ0)
    ωk, ηk = 1 / aug.μ, 1 / aug.μ ^ ηpenalty
    return _augmented_lagrangian!(
        aug, c, x0, ωk, ηk; 
        lbound=lbound, ubound=ubound, μpenalty=μpenalty, ηpenalty=ηpenalty, 
        ηupdate=ηupdate, ctol=ctol, rtol=rtol, sr1tol=sr1tol, maxiter=maxiter, 
        monitor=monitor
    )
end