using Ipopt, JuMP
import JuMP.MathOptInterface
const MOI = MathOptInterface
using Plots

include("../experiment/synthetic/synthetic_btg.jl")
include("tdist_mle.jl")

using Ipopt

btg0 = load_synthetic_btg();

function ft(theta...) 
    return tdist_mle(btg0, [theta...], 0.1)
end
#nlp = tdist_mle(btg0, [5.0], 0.1)
#println("\n############# nlp is")
#display(nlp)

model = Model(Ipopt.Optimizer)

#lx = hcat([0.01 for i = 1:1], [-5])
#ux = hcat([100 for i = 1:1], [5])

lx = [0.1 for i = 1:1]
ux = [100 for i = 1:1]
register(model, :fun, 1, ft, autodiff = true)

@variable(model, lx[i] <= au[i=1:1] <= ux[i])
set_start_value(au[1], 2.0)

@NLobjective(model, Min, fun(au...))

JuMP.optimize!(model)

