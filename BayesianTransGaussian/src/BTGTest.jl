
k = SquaredExponential(Uniform(0, 2))
θ = getparam(k, (0.5,))

g = BoxCox(Uniform(-1, 1))
λ = getparam(g, (0.5,))

println(k(θ, 0))
println(g(λ, exp(1)))

X0 = reshape(collect(1.0:5.0), 5, 1)
Y0 = exp.(X0)

println(g.(λ, Y0))

function matrixprint(x)
    (m, n) = size(x)
    for i in 1:m
        for j in 1:n
            print(floor(x[i, j] * 1000) / 1000, "      ")
        end
        println()
    end
end

matrixprint(kernelmatrix(k, θ, X0, X0))

