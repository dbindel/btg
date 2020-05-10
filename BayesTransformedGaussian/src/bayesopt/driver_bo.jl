include("../btg.jl")
include("btgBayesopt.jl")
include("../computation/finitedifference.jl")
using Plots
using Random
Random.seed!(8);
#
# trying out diff random seeds:
# failure: (3)
# success : (4, 5 works with good initialization, 6 works with init, 7 works with and w/o init.)
#use 12 for all prev experiments

#####
##### Define optimization problem, initialize GP with burn-in points, initialize btg parameters
#####

Himmelblau(x) = (x[1]^2 + x[2] -11)^2 + (x[1]+x[2]^2-7)^2+400 #function to optimize
y, x = sample_points(Himmelblau, [-5, -5], [5, 5]; num = 100) #burn-in points which are used as training data in GP
x = vcat(x, [4.999 4.999])
y = vcat(y, 1289.0763259560015)
x = vcat(x, [-4.999 4.999])
y = vcat(y, 929.3482659600018)
x = vcat(x, [4.999 -4.999])
y = vcat(y, 1009.3322659600017)
x = vcat(x, [5.0 5])
y = vcat(y, 1290.0)

@info "y-vals", y[1:5] 
### Set BTG parameters and get function handles for pdf, cdf, dpdf, etc. 
Fx = linear_polynomial_basis(x)
train = trainingData(x, Fx, y) #training data triple: location, covariates, labels
#rangeθ = reshape(select_single_theta_range(x), 1, 2) #rangetheta is auto-selected
#rangeθ = [0.0001 1000]
rangeθ = [300.0 1500] 
@info "rangeθ", rangeθ 
rangeλ = [-1.0 1.0] 
btg1 = btg(train, rangeθ, rangeλ);
(pdf, cdf, dpdf, cdf_gradient, cdf_hessian) = solve(btg1; derivatives = true) #get function handles for pdf, cdf, derivtives

#####
##### Define acquisition function and derivatives
#####

println("defining acquisition function...")
#lx = [1, -5, -5]; ux = [3000.0, 5, 5] #box-constraints for optimization problem
lx = [1, -5, -5]; ux = [3000.0, 5, 5] #box-constraints for optimization problem
#make input an augmented vector [y, s]: label, location
cdf_fixed(v) = cdf(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
cdf_gradient_fixed(v) = cdf_gradient(v[2:end], linear_polynomial_basis(v[2:end]), v[1])

d = length(lx)
fun(x...) = x[1] #x\in R^d+1 
dfun(g, x...) = (g[1] = 1; g[i] = 0 for i = 2:length(g);) 
#cdf_wrapper(x) = cdf_fixed(collect(x)) 
cdf_wrapper(x...) = cdf_fixed(collect(x))
#cdf_gradient_wrapper(g, x...) = (grad = cdf_gradient_fixed(collect(x...)); g[i] = grad[i] for i = 1:length(grad); @info grad; @info g; return nothing)
function cdf_gradient_wrapper(g, x...)
    grad = cdf_gradient_fixed(collect(x))
    @info "grad", grad
    for i = 1:length(grad)
        #@info "i", i
        #@info "grad component i", grad[i]
        global g[i] = grad[i]
    end
    return nothing
end
model = Model(Ipopt.Optimizer)
set_optimizer_attributes(model, "tol" => 1e-2, "max_iter" => 200)
#model = Model(Gurobi.Optimizer)
#NLoptSolver(algorithm=:SLSQP))
register(model, :fun, 3, fun, dfun)
register(model, :cdf_wrapper, 3, cdf_wrapper, cdf_gradient_wrapper)
#####
##### Initial guess for IPopt. Check gradient before starting optimizer
#####
#println("checking gradient at initial point...")
#####
##### Register variables and constraints in model and run optimizer.
#####
#@variable(model, lx[i] <= au[i=1:3] <= ux[i], start = initval[i])
@variable(model, lx[i] <= au[i=1:3] <= ux[i])
#@variable(model, au)
@NLobjective(model, Min, fun(au...))
@NLconstraint(model, cdf_wrapper(au...) == 0.25)

function single_optimization()
    initval = init_constrained_pt(cdf_fixed, lx, ux; quantile = 0.25)
    set_start_value(au[1], initval[1])
    set_start_value(au[2], initval[2])
    set_start_value(au[3], initval[3])
    #println("Running Optimizer...")
    #initval = [200, 4, 4]              
    #@info "initval", initval
    #@variable(model, lx[i] <= au[i=1:3] <= ux[i])
    JuMP.optimize!(model)
    println(value.(au))
    vstar = value.(au)
    return (vstar, initval)
end

include("test_script.jl")
function run_func(n)
    x = [];
    vstars = [];
    init_val_cdf = []
    init_vals = []
    for i = 1:n
        Random.seed!(i+200)
        (vstar, initval) = single_optimization();
        push!(x, cdf_fixed(vstar));
        push!(vstars, vstar);
        push!(init_val_cdf, cdf_fixed(initval))
        push!(init_vals, initval)
    end
    return x, vstars, init_val_cdf, init_vals
end

(res, vstars, init_val_cdf, init_vals) = run_func(8);
vcdfplot_sequence(vstars; res = res, upper = 2000) #shows cdfs at points we converged to

### check derivatives after the fact

#for i = 1:length(vstars)
#    check
#end

#(_, _, plt1, pol) = checkDerivative(cdf_fixed, cdf_gradient_fixed, vstars[1],  nothing, 4, 8)

for i = 1:10 #take 10 BO steps
    #(u_star, s_star) = optimize_acquisition(cdf, cdf_gradient, cdf_hessian)
    #update BTG trainingData and trainingBuffers with new point (xstar, Fxstar, ystar)  #location, covariates, label
    # update
    #update!(btg, s_star, cov_fun(s_star), u_star
    # - Σθ_inv_X, check  compute directly in O(n^2p)
    # - choleskyΣθ, check compute directly, have cholesky factorization anyways 
    # - choleskyXΣX, check compute directly
    # - logdetΣθ    compute directly
    # - logdetXΣX compute directly 
    # 
end
#4.51690700362799, 3.011185163108001
