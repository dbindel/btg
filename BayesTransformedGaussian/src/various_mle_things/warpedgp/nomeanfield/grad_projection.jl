using LinearAlgebra

mutable struct SR1Hessian
    H::Matrix{Float64}
    x::Vector{Float64}
    g::Vector{Float64}
    utol::Float64
    function SR1Hessian(n, c; utol=1e-8)
        H0 = Matrix(c * I, n, n)
        x0 = zeros(n)
        g0 = zeros(n)
        return new(H0, x0, g0, utol)
    end
end

function update!(sr1::SR1Hessian, x, g)
    s = mul!(sr1.x, x, 1, 1, -1)
    y = mul!(sr1.g, g, 1, 1, -1)

    v = mul!(y, sr1.H, s, -1, 1)
    c = dot(v, s)
    if abs(c) > sr1.utol * norm(s) * norm(v)
        mul!(sr1, v, v', 1 / c, 1)
    end

    copyto!(sr1.x, x)
    copyto!(sr1.g, g)

    return sr1
end

struct CauchyPoint
    xc::Vector{Float64}
    p::Vector{Float64}
    t::Vector{Float64}
    idx::Vector{Int}
    Gp::Vector{Float64}
end

breakpoints(x, p, l, u) = p < 0 ? (x - u) / p : p > 0 ? (x - l) / p : Inf

function _cauchy_point!(c, gx, Hx, ti_m1)
    for i in c.idx
        δt = c.t[i] - ti_m1 
        if δt != 0
            mul!(c.Gp, Hx, c.p)
            fc′ = dot(c.p, gx) + dot(c.xc, c.Gp)
            fc′′ = dot(c.p, c.Gp)
            if fc′ > 0
                return nothing
            end 
            Δt = -fc′ / fc′′
            if Δt < δt
                mul!(c.xc, c.p, 1, Δt, 1)
                return nothing
            end
            mul!(c.xc, c.p, 1, δt, 1)
            ti_m1 = c.t[i]
        end
        c.p[i] = 0
    end
    return nothing
end

function cauchy_point!(c, gx, Hx, lbound, ubound)
    copyto!(c.p, gx)
    mul!(c.p, Hx, c.xc, 1, 1)
    c.t .= breakpoints.(c.xc, c.p, lbound, ubound)
    idx = sortperm!(c.idx, c.t)
    _cauchy_point!(c, gx, Hx, 0)
    return c
end

function gradient_projection(
    f, x0, lbound, ubound; rtol
)

end

function trust_region_gp(
    f, x0;
    lbound=fill(-Inf, size(x0)), ubound=fill(Inf, size(x0)), 
    maxiter=100, rtol=1e-6, Δmax=Inf
)
    
end
