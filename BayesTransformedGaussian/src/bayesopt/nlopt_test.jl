using NLopt

function myfunc(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5/sqrt(x[2])
    end
    return sqrt(x[2])
end

function myconstraint(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

opt = NLoptSolver(;algorithm = :NLOPT_LD_SLSQP)
lower_bounds!(opt, -Inf)

opt.lower_bounds = [-Inf, 0.]
opt.xtol_rel = 1e-4

opt.min_objective = myfunc
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

(minf,minx,ret) = optimize(opt, [1.234, 5.678])
numevals = opt.numevals # the number of function evaluations
println("got $minf at $minx after $numevals iterations (returned $ret)")