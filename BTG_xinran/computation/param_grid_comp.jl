using LinearAlgebra

include("quadgrid.jl")

function param_grid_comp(n_param, param_priorInfo)
    # extract param info
    thetamin = param_priorInfo.theta.min
    thetamax = param_priorInfo.theta.max
    lambdamin = param_priorInfo.lambda.min
    lambdamax = param_priorInfo.lambda.max
    n_theta = n_param.theta
    n_lambda = n_param.lambda
   
   # form a 2d theta grid
   theta1_grid, wt1_theta = quadgrid(thetamin[1], thetamax[1], n_theta[1])
   theta2_grid, wt2_theta = quadgrid(thetamin[2], thetamax[2], n_theta[2])
   n_theta_totl = n_theta[1] * n_theta[2]
   theta_grid = zeros(n_theta_totl, 2)
   wt_theta = zeros(n_theta_totl)
   for i in 1:n_theta[1], j in 1:n_theta[2]
       theta_grid[j + (i-1) * n_theta[2], :] = [theta1_grid[i] theta2_grid[j]]
       wt_theta[j + (i-1) * n_theta[2]] = wt1_theta[i] * wt2_theta[j]
   end

   # form a 1d lambda grid
   lambda_grid, wt_lambda = quadgrid(lambdamin, lambdamax, n_lambda)

   thetaInfo = (grid = theta_grid, wt = wt_theta)
   lambdaInfo = (grid = lambda_grid, wt = wt_lambda)
   param_gridInfo = (theta = thetaInfo, lambda = lambdaInfo)

   return param_gridInfo
end