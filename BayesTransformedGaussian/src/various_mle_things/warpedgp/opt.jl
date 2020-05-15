function gradientdescent(f, α, ρ, c, current...; maxiter=200)
    ok = true
    i = 1
    while i < maxiter
        v, back = pullback(f, current...)
        grad = back(1.0)
        println(i)
        println("Grad: ", grad)
        println("Before: ", current)
        current = backtrack(f, α, ρ, c, current, grad, v)
        println("After: ", current)
        i += 1
    end
    return current
end

function backtrack(f, α, ρ, c, start, grad, v)
    ngrad = sum(x -> norm(x) ^ 2, grad)
    current = start .- α .* grad
    v′ = f(current...)
    while v′ > v - c * α * ngrad
        α *= ρ
        current = start .- α .* grad
        v′ = f(current...)
    end
    return current
end

function make_fg(f, unpack)
    return function(F, G, x)
        v, back = pullback(x) do x
            f(unpack(x)...)
        end
        if G != nothing
            G .= back(1.0)[1]
        end
        if F != nothing
            return v
        end
        return nothing
    end
end
