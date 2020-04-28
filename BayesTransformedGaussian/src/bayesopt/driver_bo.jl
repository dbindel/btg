include("../btg.jl")

#initialize bayesopt proxy function with burn-in points
y, x = sample(f, lx, ux)
Fx = linear_polynomial_basis(x)
train = trainingData(x, Fx, y)
rangeθ = 
rangeλ = [1, 2]
btg1 = btg(train, rangeθ, rangeλ);

for i = 1:10 #take 10 BO steps
    #(pdf, cdf, dpdf, cdf_gradient, cdf_hessian) = solve(btg)
    #(u_star, s_star) = optimize_acquisition(cdf, cdf_gradient, cdf_hessian)
    #update BTG trainingData and trainingBuffers with new point (xstar, Fxstar, ystar)  #location, covariates, label
    # update
    update!(btg, s_star, cov_fun(s_star), u_star)

    # - Σθ_inv_X, check  compute directly in O(n^2p)
    # - choleskyΣθ, check compute directly, have cholesky factorization anyways 
    # - choleskyXΣX, check compute directly
    # - logdetΣθ    compute directly
    # - logdetXΣX compute directly
    # 
    #
end
"""
cdf(u, s) is a function of location s and value u, gradients and hessian are
w.r.t the vector (u, s).
"""
function optimize_acquisition_func(cdf, cdf_gradient, cdf_hessian)
    
    
    #call optim routine, to get vector (u_star, s_star)
end
