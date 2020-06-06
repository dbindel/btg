using Ipopt, JuMP
import JuMP.MathOptInterface
const MOI = MathOptInterface
using Plots

#include("../experiment/synthetic/synthetic_btg.jl")
include("tdist_mle.jl")

using Ipopt

function optimize_lambda(fun, lambda_range)
    model = Model(Ipopt.Optimizer)
    #lx = hcat([0.01 for i = 1:1], [-5])
    #ux = hcat([100 for i = 1:1], [5])
    lx = [lambda_range[1]]
    ux = [lambda_range[2]]
    register(model, :fun, 1, fun, autodiff = true)
    @variable(model, lx[1] <= au[i=1:1] <= ux[1])
    set_start_value(au[1], 0.0001)
    @NLobjective(model, Min, fun(au[1]))
    JuMP.optimize!(model)
    return value.(au)
end

"""
MLE estimate for single dimensional theta and lambda.
    INPUTS: 
        - search bounds
        - 2-variable function fun
    THIS FUNCTION GIVES A WEIRD ERROR.
"""
function optimize_theta_lambda_single_broken(fun, theta_range, lambda_range)
    model = Model(Ipopt.Optimizer)
    lx = [theta_range[1], lambda_range[1]]
    ux = [theta_range[2], lambda_range[2]]
    register(model, :func, 2, fun, autodiff = true)
    @variable(model, lx[i] <= au[i=1:2] <= ux[i])
    set_start_value(au[1], 100.0)
    set_start_value(au[2], 0.001)
    @NLobjective(model, Min, func(au[1], au[2]))
    JuMP.optimize!(model)
    return value.(au)
end

using Optim
function optimize_theta_lambda_single(fun, theta_range, lambda_range)
    x0 = [100.0, 0.01]
    optimize(fun, x0)
end