using BenchmarkTools, LinearAlgebra

include("./kernel.jl")

function blarg()
    U = rand(3, 10)
    X = rand(3, 500)
    Fu = lu!(rand(10, 10))
    Fx = rand(10, 500)
    W = Fu \ Fx

    k = FixedParam(Gaussian(), 0.1)

    Ku = correlation(k, U; jitter = 0)
    Kux = cross_correlation(k, U, X)
    Kx = correlation(k, X; jitter = 0)
    function g1()
        cholesky(Symmetric(Kx .- W' * Kux .- Kux' * W .+ W' * Ku * W))
        return nothing
    end
    function g2()
        c = cholesky(Symmetric(Kx))
        f = c.L \ Fx'
        cholesky!(f' * f)
        return nothing
    end
    println("Got here")
    b1 = @benchmark $g1()
    println("And here")
    b2 = @benchmark $g2()
    return b1, b2
end
