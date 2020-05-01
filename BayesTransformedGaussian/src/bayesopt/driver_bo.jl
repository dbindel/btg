include("../btg.jl")
include("btgBayesopt.jl")

using Random
Random.seed!(0);
#initialize bayesopt proxy function with burn-in points
Himmelblau(x) = (x[1]^2 + x[2] -11)^2 + (x[1]+x[2]^2-7)^2 +1
y, x = sample(Himmelblau, [-5, -5], [5, 5]; num = 20)
@info "y-vals", y[1:5] 
Fx = linear_polynomial_basis(x)
train = trainingData(x, Fx, y)
rangeθ = reshape(select_single_theta_range(x), 1, 2)
@info "rangeθ", rangeθ
rangeλ = [1 5.0]
btg1 = btg(train, rangeθ, rangeλ);
(pdf, cdf, dpdf, cdf_gradient, cdf_hessian) = solve(btg1; derivatives = true)
println("optimizing acquisition function...")
lx = [1, -5, -5]; ux = [2000.0, 5, 5]
(vstar, mini) = optimizeUCB(cdf, cdf_gradient, cdf_hessian, lx, ux);
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
