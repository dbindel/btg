
using LinearAlgebra

include("../param/param_prior.jl")
include("../kernels/kernel.jl")
include("../kernels/mean_basis.jl")
include("../transform/nonlinfun.jl")
 
function weight_comp(param_gridInfo, param_priorInfo, trainBasicInfo)
    # extract training data info
    n = size(trainBasicInfo.data[2], 1) # number of training points
    
    # extract param grid and pdf info
    theta_grid = param_gridInfo.theta.grid
    wt_theta = param_gridInfo.theta.wt
    n_theta = length(wt_theta)
    pdf_theta = param_priorInfo.theta.pdf

    lambda_grid = param_gridInfo.lambda.grid
    wt_lambda = param_gridInfo.lambda.wt
    n_lambda = length(wt_lambda)
    pdf_lambda = param_priorInfo.lambda.pdf

    # initiate and pre-allocate space
    alpha0 = zeros(n_theta, n_lambda)
    L_Set = zeros(n_theta, n_lambda, n, n)
    # should be p*p, but p = 1, so treat XSX as a scalar for now
    XSX_Set = zeros(n_theta, n_lambda) 
    gz_Set = zeros(n_theta, n_lambda, n)
    q_Set = zeros(n_theta, n_lambda)
    # Beta_Set = zeros(n_theta, n_lambda, p)
    Beta_Set = zeros(n_theta, n_lambda)

    for i in 1:n_theta, j in 1:n_lambda
        h_ij, trainingInfo = hfun(theta_grid[i, :], lambda_grid[j], trainBasicInfo)
        h_ij = h_ij * pdf_theta(theta_grid[i, :]) * pdf_lambda(lambda_grid[j])
        alpha0[i, j] = wt_theta[i] * wt_lambda[j] * h_ij
        L_Set[i, j, :, :] = trainingInfo.L
        XSX_Set[i, j] = trainingInfo.XSX
        gz_Set[i, j, :] = trainingInfo.gz
        q_Set[i, j] = trainingInfo.q
        Beta_Set[i, j] = trainingInfo.Beta
    end

    # normalize alpha_ij
    C = sum(alpha0)
    alpha = alpha0 ./ C
   

    sideSetInfo = (L = L_Set, XSX = XSX_Set, 
                    gz = gz_Set, q = q_Set, Beta = Beta_Set)

    return alpha, sideSetInfo
end


# function h = p(z|eta)p(eta), 
# after integrating out beta and tau
# RHS of equation (8)
function hfun(theta, lambda, trainBasicInfo)
    x = trainBasicInfo.data[1]
    z = trainBasicInfo.data[2]
    g = trainBasicInfo.nonlintrans.fun
    dg = trainBasicInfo.nonlintrans.deriv
    kernel = trainBasicInfo.kernel
    p = trainBasicInfo.polydof

    X = meanbasisMat(x, p)
    n = size(x, 1)
    @assert size(x, 1) == size(X, 1)

#= OLD VERSION
    Sigma = KernelMat(x, x, kernel, theta)
    L_Sigma = cholesky(Sigma).U # L' * L = S
    invL = inv(L_Sigma)
    invSigma = invL * invL'
    XSigmaX = X' * Sigma * X
    invXSigmaX = inv(XSigmaX)
    gz = g.(z, lambda)
    Beta = invXSigmaX * X' * invSigma * gz
    gzminusXbeta = gz - X*Beta
    q = gzminusXbeta' * invSigma * gzminusXbeta
    dg_new = z -> dg(z, lambda)
    J = DetJ(z, dg_new)
    h = det(invL) * det(XSigmaX)^(-1/2) * q^((p - n)/2) * J^(1-p/n)
 =#

    Sigma = KernelMat(x, x, kernel, theta)
    L_Sigma = cholesky(Sigma).L # L * L' = S
    XSigmaX = dot(L_Sigma\X, L_Sigma\X)
    gz = g.(z, lambda)
    Beta = XSigmaX \ X' * (L_Sigma' \ (L_Sigma \ gz)) 
    q_half = L_Sigma \ (gz - X*Beta)
    q = dot(q_half, q_half)
    dg_new = z -> dg(z, lambda)
    J = DetJ(z, dg_new)
    h = (1/det(L_Sigma)) * sqrt(1/det(XSigmaX)) * sqrt(q^(p - n)) * J^(1-p/n)

    trainingInfo = (L = L_Sigma, XSX = XSigmaX, 
                gz = gz, q = q, Beta = Beta)
    return h, trainingInfo
end
