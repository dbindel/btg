include("../btg.jl")
include("btgBayesopt.jl")
include("../computation/finitedifference.jl")
using Plots
using Random
Random.seed!(0);

#####
##### Define optimization problem, initialize GP with burn-in points, initialize btg parameters
#####

Himmelblau(x) = (x[1]^2 + x[2] -11)^2 + (x[1]+x[2]^2-7)^2+400 #function to optimize
y, x = sample_points(Himmelblau, [-5, -5], [5, 5]; num = 12) #burn-in points which are used as training data in GP
@info "y-vals", y[1:5] 
### Set BTG parameters and get function handles for pdf, cdf, dpdf, etc. 
Fx = linear_polynomial_basis(x)
train = trainingData(x, Fx, y) #training data triple: location, covariates, labels
rangeθ = reshape(select_single_theta_range(x), 1, 2) #rangetheta is auto-selected
@info "rangeθ", rangeθ
rangeλ = [-1.0 1.0] 
btg1 = btg(train, rangeθ, rangeλ);
(pdf, cdf, dpdf, cdf_gradient, cdf_hessian) = solve(btg1; derivatives = true) #get function handles for pdf, cdf, derivtives

#####
##### Define acquisition function and derivatives
#####

println("defining acquisition function...")
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
#model = Model(Gurobi.Optimizer)
#NLoptSolver(algorithm=:SLSQP))
register(model, :fun, 3, fun, dfun)
register(model, :cdf_wrapper, 3, cdf_wrapper, cdf_gradient_wrapper)

#####
##### Initial guess for IPopt. Check gradient before starting optimizer
#####

#println("checking gradient at initial point...")
initval = init_constrained_pt(cdf_fixed, lx, ux; quantile = 0.25)
#store = [0.0 0 0]
#(r1, r2, plt1, pol1) = checkDerivative_in_place(cdf_wrapper, cdf_gradient_wrapper, initval, store, nothing); #input here is a vector

#####
##### Register variables and constraints in model and run optimizer.
#####
if true
    println("Running Optimizer...")
    #initval = [200, 4, 4]              
    @info "initval", initval
    @variable(model, lx[i] <= au[i=1:3] <= ux[i], start = initval[i])
    @NLobjective(model, Min, fun(au...))
    @NLconstraint(model, cdf_wrapper(au...) == 0.25)
    JuMP.optimize!(model)
    println(value.(au))
    vstar = value.(au)
end
include("test_script.jl")

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
