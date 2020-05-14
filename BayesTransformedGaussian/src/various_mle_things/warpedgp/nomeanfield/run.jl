using LinearAlgebra, Zygote, Optim, Arpack
using CSV, DataFrames

include("model.jl")


function data()
    abalone = CSV.File(open("abalone.csv"))
    abalone = DataFrame(abalone)
    X = Matrix(abalone[1:200,2:8])
    y = Vector{Float64}(abalone[1:200, 9])
    return X, y
end

function testfun()
    X, y = data()
    g = SumTanh()
    k = Gaussian()
    mdl = Model(X, y, k, g, 1)
    unpack = x -> (x[1:7], x[8], x[9], x[10:11], x[12:13], x[14:15])
    fg! = make_fg(unpack) do ℓ, ϵ, amp, α, β, c
       -logprob(mdl, Diagonal(ℓ), (amp,), ϵ, (α, β, c))
    end
    return Optim.optimize(Optim.only_fg!(fg!), rand(15) / 5, LBFGS())
end
