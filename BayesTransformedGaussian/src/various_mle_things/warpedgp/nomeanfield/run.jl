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
    g = DoNothing()
    k = Gaussian()
    mdl = Model(X, y, k, g, 1)
    unpack = x -> @views (x[1], x[2], x[3])
    fg! = make_fg(unpack) do ℓ, ϵ, amp
       -logprob(mdl, UniformScaling(ℓ), (amp,), ϵ, ())
    end
    return Optim.optimize(Optim.only_fg!(fg!), rand(3) .+ 5, LBFGS())
end
