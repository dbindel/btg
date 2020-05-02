include("../covariatefun.jl")

using Ipopt, JuMP
import JuMP.MathOptInterface
const MOI = MathOptInterface

function opt_UCB(cdf, lx, ux)
    cdf_fixed(v) = cdf(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
    d = length(lx)
    fun(x) = x[1] #x\in R^d+1 
    #cdf_wrapper(x) = cdf_fixed(collect(x)) 
    cdf_wrapper(x) = cdf_fixed(x)
    model = Model(Ipopt.Optimizer)
    register(model, :constraint, 3, cdf_wrapper; autodiff = true)
    register(model, :objective, 3, fun; autodiff = true)
    @variable(model, lx[i] <=au[i=1:3] <= ux[i])
    @NLobjective(model, Min, objective(au))
    @NLconstraint(model, constraint(au) == 0.25)
    JuMP.optimize!(model)
    return value.(model)
end