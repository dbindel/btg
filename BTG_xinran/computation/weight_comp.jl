
using LinearAlgebra

include("param_prior.jl")
include("kernel.jl")
include("mean_basis.jl")
include("nonlinfun.jl")

function weight_comp(param_gridInfo, param_priorInfo, trainBasicInfo)
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
    invS_Set = zeros(n_theta, n_lambda, n, n)
    # invXSigmaX_Set = zeros(n_theta, n_lambda, p, p)
    invXSX_Set = zeros(n_theta, n_lambda)
    gz_Set = zeros(n_theta, n_lambda, n)
    q_Set = zeros(n_theta, n_lambda)
    # Beta_Set = zeros(n_theta, n_lambda, p)
    Beta_Set = zeros(n_theta, n_lambda)

    for i in 1:n_theta, j in 1:n_lambda
        h_ij, trainingInfo = hfun(theta_grid[i, :], lambda_grid[j], trainBasicInfo)
        h_ij = h_ij * pdf_theta(theta_grid[i, :]) * pdf_lambda(lambda_grid[j])
        alpha0[i, j] = wt_theta[i] * wt_lambda[j] * h_ij
        invS_Set[i, j, :, :] = trainingInfo.invS
        invXSX_Set[i, j] = trainingInfo.invXSX
        gz_Set[i, j, :] = trainingInfo.gz
        q_Set[i, j] = trainingInfo.q
        Beta_Set[i, j] = trainingInfo.Beta
    end

    # normalize alpha_ij
    C = sum(alpha0)
    alpha = alpha0 ./ C

    sideSetInfo = (invS = invS_Set, invXSX = invXSX_Set, 
                    gz = gz_Set, q = q_Set, Beta = Beta_Set)

    return alpha, sideSetInfo
end


# function h = p(z|eta)p(eta), 
# after integrating out beta and tau
function hfun(theta, lambda, trainBasicInfo)
    x = trainBasicInfo.traindata.idx
    z = trainBasicInfo.traindata.val
    g = trainBasicInfo.nonlintrans.fun
    dg = trainBasicInfo.nonlintrans.deriv
    kernel = trainBasicInfo.kernel
    p = trainBasicInfo.polydof

    X = meanbasisMat(x, p)
    n = size(x, 1)
    @assert size(x, 1) == size(X, 1)

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
    trainingInfo = (invS = invSigma, invXSX = invXSigmaX, 
                gz = gz, q = q, Beta = Beta)
    return h, trainingInfo
end
