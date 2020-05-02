include("../btg.jl")
include("btgBayesopt.jl")
using Random
Random.seed!(0);
##
## initialize bayesopt proxy function with burn-in points
##
Himmelblau(x) = - (x[1]^2 + x[2] -11)^2 + (x[1]+x[2]^2-7)^2 + 3000
y, x = sample_points(Himmelblau, [-5, -5], [5, 5]; num = 150) #burn-in points which are used as training data in GP
@info "y-vals", y[1:5] 
### Set BTG parameters and get function handles for pdf, cdf, dpdf, etc. 
Fx = linear_polynomial_basis(x)
train = trainingData(x, Fx, y)
rangeθ = reshape(select_single_theta_range(x), 1, 2)
@info "rangeθ", rangeθ
rangeλ = [-1.0 1.0]
btg1 = btg(train, rangeθ, rangeλ);
(pdf, cdf, dpdf, cdf_gradient, cdf_hessian) = solve(btg1; derivatives = true)

###
println("optimizing acquisition function...")
lx = [1, -5, -5]; ux = [6000.0, 5, 5]
#(vstar, mini, res) = optimizeUCB(cdf, cdf_gradient, cdf_hessian, lx, ux);
cdf_fixed(v) =(v = collect(v); cdf(v[2:end], linear_polynomial_basis(v[2:end]), v[1]))
cdf_fixed_arr(v) = cdf(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
cdf_gradient_fixed(v) = (v = collect(v); cdf_gradient(v[2:end], linear_polynomial_basis(v[2:end]), v[1]))
d = length(lx)
fun(x...) = x[1] #x\in R^d+1 
dfun(g, x...) = (g[1] = 1; g[i] = 0 for i = 2:length(g);) 
#cdf_wrapper(x) = cdf_fixed(collect(x)) 
cdf_wrapper(x...) = cdf_fixed(x)
cdf_gradient_wrapper(g, x...) = (grad = cdf_gradient_fixed(x); g[i] = grad[i] for i =1:length(grad);)
model = Model(Ipopt.Optimizer)
#model = Model(Gurobi.Optimizer)
#NLoptSolver(algorithm=:SLSQP))
register(model, :fun, 3, fun, dfun)
register(model, :cdf_wrapper, 3, cdf_wrapper, cdf_gradient_wrapper)
#register(model, :fun, 3, fun, autodiff = true)
#register(model, :cdf_wrapper, 3, cdf_wrapper, autodiff = true)
initval = init_constrained_pt(cdf_fixed_arr, lx, ux; quantile = 0.75)
#initval = [200, 4, 4]              
@info "initval", initval
@variable(model, lx[i] <= au[i=1:3] <= ux[i], start = initval[i])
@NLobjective(model, Max, fun(au...))
@NLconstraint(model, cdf_wrapper(au...) == 0.75)
#NLconstraint(model, cdf_wrapper(au...) = 0.23)
#@NLconstraint(model, cdf_wrapper(au...) <= 0.27)
JuMP.optimize!(model)
println(value.(au))
vstar = value.(au)
include("test_script.jl")
#vstar = opt_UCB(cdf, lx, ux)
#@info "vstar", vstar
#@info "minimum", minimum
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
