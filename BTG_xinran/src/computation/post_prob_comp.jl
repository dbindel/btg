
# compute p(z0|z)
function pdf_z0(x0, z0, trainBasicInfo, sideSetInfo, param_gridInfo, alpha, Gamma)
    # extract 
    theta_grid = param_gridInfo.theta.grid
    lambda_grid = param_gridInfo.lambda.grid
    L_Set = sideSetInfo.L
    XSX_Set = sideSetInfo.XSX
    gz_Set = sideSetInfo.gz
    q_Set = sideSetInfo.q
    Beta_Set = sideSetInfo.Beta

    n_theta = size(theta_grid, 1)
    n_lambda = size(lambda_grid, 1)
    
    p_z0 = 0.0
    
    for i in 1:n_theta, j in 1:n_lambda 
        tempsideInfo = (L = L_Set[i, j, :, :], 
                    XSX = XSX_Set[i, j, :, :], 
                    gz = gz_Set[i, j, :], 
                    q = q_Set[i, j], 
                    Beta = Beta_Set[i, j, :])
        p_z0_sample = pdf_z0_ij(theta_grid[i, :], lambda_grid[j], x0, z0,
                                trainBasicInfo, tempsideInfo, Gamma)
        p_z0 += alpha[i, j] * p_z0_sample 
    end
    
    return p_z0

end


function pdf_z0_ij(theta_sample, lambda_sample, x0, z0, trainBasicInfo, sideInfo, Gamma)
    x = trainBasicInfo.data[1]
    z = trainBasicInfo.data[2]
    g = trainBasicInfo.nonlintrans.fun
    dg = trainBasicInfo.nonlintrans.deriv
    kernel = trainBasicInfo.kernel
    p = trainBasicInfo.polydof

    X = meanbasisMat(x, p) # n * p
    X0 = meanbasisMat(x0, p) # k * p
    n = size(x, 1)
    k = size(x0, 1)

    L = sideInfo.L # n * n
    XSigmaX = sideInfo.XSX # p * p
    gz = sideInfo.gz # n * 1
    q = sideInfo.q # 1 * 1
    Beta = sideInfo.Beta # p * 1

    B = KernelMat(x0, x, kernel, theta_sample) # k * n
    E = KernelMat(x0, x0, kernel, theta_sample) # k * k

    if (k == 1) && (p == 1)
    
        # compute m
        BinvS = (B / L') / L # k * n
        H = X0[1] - (BinvS * X)[1] # k * p 
        m = (BinvS * gz)[1] + H * Beta[1]
        # compute C 
        D = E[1] - (BinvS * B')[1]
        C = D + H / (XSigmaX[1]) * H

    else
    

        # compute m
        BinvS = (B / L') / L # k * n
        H = X0 - BinvS * X # k * p 
        m = BinvS * gz + H * Beta
        # compute C 
        D = E - BinvS * B'
        C = D + H / XSigmaX * H'
    end

    # cpmpute p(z0|theta, lambda, z)
    dg_new = z -> dg(z, lambda_sample)
    gz0 = g(z0, lambda_sample) - m 
    p_z0_sample = Gamma * sqrt(( 1+ (gz0'/C*gz0)/q ) ^(p-n-k)) * abs(dg_new(z0)) * 1/sqrt(q*det(C))

    return p_z0_sample
end