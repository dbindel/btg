using LinearAlgebra, Zygote, Optim, IterativeSolvers, Arpack
using CSV, DataFrames

include("model.jl")

function data()
    abalone = CSV.File(open("abalone.csv"))
    abalone = DataFrame(abalone)
    X = Matrix(Matrix(abalone[1:200,2:8])')
    y = Vector(abalone[1:200, 9])
    f = Linear()
    g = SumTanh()
    k = Gaussian()
    d = data(f, X, y)
    return f, g, k, d
end

function testfun()
    f, g, k, d = data()
    unpack = x -> (x[1:5], x[6], x[7], x[8:9], x[10:11], x[12:13])
    fg! = make_fg(unpack) do ℓ, ϵ, amp, α, β, c
       ks = kernelsystem(f, k, g, d, Diagonal(ℓ), (amp,), ϵ, (α, β, c))
       -logprob(ks, (α, β, c), 0)
    end
    return Optim.optimize(Optim.only_fg!(fg!), rand(13), ConjugateGradient())
end

function datavanilla()
    abalone = CSV.File(open("abalone.csv"))
    abalone = DataFrame(abalone)
    X = Matrix(Matrix(abalone[1:200,2:6])')
    y = Vector(abalone[1:200, 7])
    f = Linear()
    g = Identity()
    k = Gaussian()
    d = data(f, X, y)
    return f, g, k, d
end

function testfunvanilla()
    f, g, k, d = datavanilla()
    unpack = x -> (x[1:5], x[6], x[7])
    fg! = make_fg(unpack) do ℓ, ϵ, amp
       ks = kernelsystem(f, k, g, d, Diagonal(ℓ), (amp,), ϵ, ())
       -logprob(ks, (), 0)
    end
    return Optim.optimize(Optim.only_fg!(fg!), rand(7), ConjugateGradient())
end

function datavanillanoτ()
    abalone = CSV.File(open("abalone.csv"))
    abalone = DataFrame(abalone)
    X = Matrix(Matrix(abalone[1:200,2:6])')
    y = Vector(abalone[1:200, 7])
    f = Linear()
    g = Identity()
    k = Gaussian()
    d = data(f, X, y)
    return f, g, k, d
end

function testfunvanillanoτ()
    f, g, k, d = datavanillanoτ()
    unpack = x -> (x[1:5], x[6])
    fg! = make_fg(unpack) do ℓ, ϵ
       ks = kernelsystem(f, k, g, d, Diagonal(ℓ), (1 / (1 + ϵ),), ϵ / (1 + ϵ), ())
       -logprob(ks, (), 0)
    end
    return Optim.optimize(Optim.only_fg!(fg!), rand(6), ConjugateGradient())
end
