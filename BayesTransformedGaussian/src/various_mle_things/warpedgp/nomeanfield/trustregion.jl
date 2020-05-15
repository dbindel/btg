using LinearAlgebra

function template!(g, H, x)
end

function trs_hard!(g, cH::Cholesky, Δ, v)
    n = length(g)
    v′ = v .* sqrt(norm(cH)) ./ n
    lowrankupdate!(cH, v′)
    q = ldiv!(cH, g)
    return v .* sqrt(Δ ^ 2 - dot(q, q)) .- q
end

function trs_hard!(g, H::AbstractMatrix, Δ, v)
    n = length(g)
    H′ = mul!(H, v, v', norm(H) / n ^ 2, 1)
    q = ldiv!(H′, g)
    return v .* sqrt(Δ ^ 2 - dot(q, q)) .- q
end

function trs_eig!(trA, g, H; Δ, gaptol=eps()*10^8)
    n = length(g)

    Λ, V = eigen(trA)
    λ1, λ2 = float(Λ[1]), float(Λ[2])
    y2 = float.(V[n+1:end, 1])
    y1 = float.(V[1:n, 1])

    if norm(y1) <= gaptol / sqrt(λ1 - λ2)
        v = y2 / norm(y2)
        return trs_hard!(g, H, Δ, v)
    else
        return -sign(dot(g, y2)) .* Δ .* y1 ./ norm(y1)
    end
end

function trust_region_sub!(g, H; Δ, gaptol=eps()*10^8)
    n = length(g)

    w = g ./ Δ
    trA = [H Matrix(1.0 * I, n, n); w * w' H]

    cH = cholesky(H; check=false)
    if issuccess(cH)
        p = cH \ g
        if norm(p) <= Δ
            return p, false
        else
            return trs_eig!(trA, g, cH; Δ=Δ, gaptol=gaptol), true
        end
    end
    return trs_eig!(trA, g, H; Δ=Δ, gaptol=gaptol), true
end

function _tr_interp!()
end

function trust_region_newton(
    x0, feval!; 
    maxiter=100, Δmax=Inf, monitor=(x, rnorm, Δ) -> nothing
) 
    n = length(x0)
    x = copy(x0)
    g = similar(x)
    H = similar(x, n, n)

    f = feval!((), g, H, x)

    cH = cholesky(H; check=false)
    nH = opnorm(H)
    p = -H \ g

    Δ = 1.2 * dot(p, p)
    active_constraint = false
    return nothing
end


