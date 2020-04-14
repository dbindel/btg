


for i = 1:10 #take 10 BO steps
    #(pdf, cdf, dpdf, cdf_gradient, cdf_hessian) = solve(btg)
    #(u_star, s_star) = optimize_acquisition(cdf, cdf_gradient, cd_hessian)
    #update BTG trainingData and trainingBuffers with new point (xstar, Fxstar, ystar)  #location, covariates, label
    # update
    # - Σθ_inv_X, check
    # - choleskyΣθ, check
    # - choleskyXΣX, check
    # - logdetΣθ    
    # - logdetXΣX
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
