include("./aug_lagrangian.jl")

function aug_test()
    f(x, ::Val{false}) = x[1] ^ 2 + x[2] ^ 2 - 2
    f(x, ::Val{true}) = x[1] ^ 2 + x[2] ^ 2 - 2, 2 .* [x[1], x[2]]
    c(x, ::Val{false}) = [x[1] - 2]
    c(x, ::Val{true}) = [x[1] - 2], [1 0;]
    x, ok = augmented_lagrangian(f, c, [10.0, 10.0], [5.0], 10.0; rtol=1e-10, maxiter=20, μpenalty=10)
    display(x)
    display(ok)
end