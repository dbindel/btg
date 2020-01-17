module btgDeriv

include("kernel.jl")
include("transforms.jl")
using Distributions
using Printf
using SpecialFunctions
using Plots
using Polynomials

export prob, partial_theta, partial_lambda, posterior_theta, checkDerivative

"""
p(z0| theta, lambda, z)
"""
function prob(θ, λ, setting)
    #@check_args (2>1)
    s = setting.s 
    s0 = setting.s0 
    X = setting.X
    X0 = setting.X0 
    z = setting.z
    g = boxCox #boxCox by default
    n = size(X, 1) 
    p = size(X, 2) 
    k = size(X0, 1) 
    Eθ = K(s0, s0, θ, rbf) 
    Σθ = K(s, s, θ, rbf) 
    Bθ = K(s0, s, θ, rbf) 
    #@printf("size of Sigma_theta: %f \n", string.(size(Σθ)))
    #@printf("size of Btheta: %f \n", string.(size(Bθ)))
    choleskyΣθ = cholesky(Σθ) 
    choleskyXΣX = cholesky(Hermitian(X'*(choleskyΣθ\X))) 
    Dθ = Eθ - Bθ*(choleskyΣθ\Bθ') 
    Hθ = X0 - Bθ*(choleskyΣθ\X) 
    Cθ = Dθ + Hθ*(choleskyXΣX\Hθ') 
    βhat = (X'*(choleskyΣθ\X))\(X'*(choleskyΣθ\g(z, λ))) 
    qtilde = (expr = g(z, λ)-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ*(cholesky(Σθ)\g(z, λ)) + Hθ*βhat 
    #t = LocationScale(m[1], sqrt(qtilde[1]*Cθ[1]/(n-p)), TDist(n-p))
    #p = z0 -> Distributions.pdf(t, g(z0, λ))
    #return p
    #cc = gamma((n-p+k)/2)/gamma((n-p)/2)/pi^(k/2) #constant term
    expr = z0 -> g(z0, λ) .- m
    return z0 -> (det(qtilde*Cθ)^(-1/2))*(1+expr(z0)'*((qtilde*Cθ)\expr(z0)))^(-(n-p+k)/2)
    
end

"""
Compute derivative of p(z0|theta, lambda, z) w.r.t theta
"""
function partial_theta(θ, λ, setting)
    #@check_args (2>1)
    s = setting.s 
    s0 = setting.s0 
    X = setting.X
    X0 = setting.X0 
    z = setting.z
    g = boxCox #boxCox by default
    n = size(X, 1) 
    p = size(X, 2) 
    k = size(X0, 1) 
    Eθ = K(s0, s0, θ, rbf) 
    Σθ = K(s, s, θ, rbf) 
    Bθ = K(s0, s, θ, rbf) 
    choleskyΣθ = cholesky(Σθ) 
    choleskyXΣX = cholesky(Hermitian(X'*(choleskyΣθ\X))) 
    Dθ = Eθ - Bθ*(choleskyΣθ\Bθ') 
    Hθ = X0 - Bθ*(choleskyΣθ\X) 
    Cθ = Dθ + Hθ*(choleskyXΣX\Hθ') 
    βhat = (X'*(choleskyΣθ\X))\(X'*(choleskyΣθ\g(z, λ))) 
    qtilde = (expr = g(z, λ)-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ*(choleskyΣθ\g(z, λ)) + Hθ*βhat 
    
    #cc = gamma((n-p+k)/2)/gamma((n-p)/2)/pi^(k/2) #constant term
    Eθ_prime = K(s0, s0, θ, rbf_prime) 
    Σθ_prime = K(s, s, θ, rbf_prime) 
    Bθ_prime = K(s0, s, θ, rbf_prime) 

    Σθ_inv_X = Σθ\X #precomputation 

    #compute betahat_prime_theta
    expr_mid = X'*(Σθ\Σθ_prime)*(Σθ_inv_X)
    AA = choleskyXΣX\(expr_mid)*(choleskyXΣX\(X'*(Σθ\g(z, λ))))
    BB = - (choleskyXΣX\(X'*(Σθ\(Σθ_prime*(Σθ\g(z, λ))))))
    βhat_prime_theta = AA + BB

    #compute qtilde_prime_theta
    vv = g(z, λ) - X*βhat
    rr = X*βhat_prime_theta
    AA = (-rr)' * (Σθ \ vv)
    BB = - vv' * (Σθ \ (Σθ_prime * (Σθ \ vv)))
    CC =  vv' * (Σθ \ (-rr))
    qtilde_prime_theta = AA + BB + CC

    #compute H_prime 
    AA = -Bθ_prime*Σθ_inv_X 
    #BB = Bθ*Σθ\(Σθ_prime*(Σθ\X)) displaying the bug
    BB = Bθ*(Σθ\(Σθ_prime*(Σθ\X)))
    H_prime = AA + BB

    #compute m_prime_theta
    AA = Bθ_prime*(Σθ\g(z, λ))
    BB = - Bθ*(Σθ\(Σθ_prime*(Σθ\g(z, λ))))
    CC = H_prime*βhat
    DD = Hθ*βhat_prime_theta
    m_prime_theta = AA + BB + CC + DD

    #compute D_theta_prime
    sigma_inv_B = Σθ \ Bθ' #precomputation
    AA = Eθ_prime - Bθ_prime * sigma_inv_B 
    BB = sigma_inv_B' * Σθ_prime * sigma_inv_B
    CC = - sigma_inv_B' * Bθ_prime'
    D_theta_prime = AA + BB + CC

    #compute C_theta_prime
    AA = D_theta_prime + H_prime*(choleskyXΣX\Hθ')
    BB = Hθ*(choleskyXΣX\(Σθ_inv_X'*Σθ_prime*Σθ_inv_X))*(choleskyXΣX\Hθ')
    CC = Hθ*(choleskyXΣX\(H_prime'))
    C_theta_prime = AA + BB + CC

    #compute derivative of main expression
    expr = z0 -> g(z0, λ) .- m
    qC = qtilde*Cθ 
    bilinearform = z0 -> 1 .+ expr(z0)'*(qC\(expr(z0)))
    qC_inv = qC\I
    detqC = det(qC) 
    qC_prime_theta =  qtilde_prime_theta*Cθ + qtilde*C_theta_prime
    AA = -0.5 * detqC^(-1/2) * tr(qC\(qC_prime_theta)) 
    qC_inv_prime_theta = - qC\(qC_prime_theta * qC_inv)
    BB = z0 -> -m_prime_theta'*(qC\(expr(z0)))
    CC = z0 -> expr(z0)' * qC_inv_prime_theta * expr(z0)
    DD = z0 -> -expr(z0)'*(qC\m_prime_theta)
    EE = z0 -> bilinearform(z0)^(-(n-p+k)/2)
    FF = z0 -> detqC^(-1/2) * (-(n-p+k)/2) * (bilinearform(z0))^(-(n-p+k+2)/2)
    
    main_deriv = z0 -> [(AA*EE(z0) + FF(z0)*(BB(z0) + CC(z0) + DD(z0)))]
    #main_deriv = (AA*EE(z0) + FF(z0)*(BB(z0) + CC(z0) + DD(z0)))
    main = z0 -> (detqC^(-1/2))*(bilinearform(z0))^(-(n-p+k)/2)

    #return (βhat, βhat_prime_theta)
    #return ([qtilde], [qtilde_prime_theta])
    #return (vec(Eθ), vec(Eθ_prime))
    #return (vec(Bθ), vec(Bθ_prime))
    #return (vec(Σθ), vec(Σθ_prime))
    #return (vec(Hθ), vec(H_prime))
    #return (m, m_prime_theta)
    #return (vec(Dθ), vec(D_theta_prime))
    #return (vec(Cθ), vec(C_theta_prime))
    return (main, main_deriv)
    #return (vec(qC_inv), vec(qC_inv_prime_theta))
    #return ([detqC^(-1/2)], [AA])
    #return (bilinearform, z0 -> BB(z0) + CC(z0) + DD(z0))
end

"""
Compute derivative of p(z0|theta, lambda, z) w.r.t lambda
"""
function partial_lambda(θ, λ, setting)
    #@check_args (2>1)
    s = setting.s 
    s0 = setting.s0 
    X = setting.X
    X0 = setting.X0 
    z = setting.z
    g = boxCox #boxCox by default
    dg = boxCoxPrime
    dgλ = boxCoxPrime_lambda
    dgλ2 = boxCoxPrime_lambda2
    dgλx = boxCoxMixed_lambda_z
    n = size(X, 1) 
    p = size(X, 2) 
    k = size(X0, 1) 
    Eθ = K(s0, s0, θ, rbf) 
    Σθ = K(s, s, θ, rbf) 
    Bθ = K(s0, s, θ, rbf) 
    choleskyΣθ = cholesky(Σθ) 
    choleskyXΣX = cholesky(Hermitian(X'*(choleskyΣθ\X))) 
    Dθ = Eθ - Bθ*(choleskyΣθ\Bθ') 
    Hθ = X0 - Bθ*(choleskyΣθ\X) 
    Cθ = Dθ + Hθ*(choleskyXΣX\Hθ') 
    βhat = choleskyXΣX\(X'*(choleskyΣθ\g(z, λ))) 
    qtilde = (expr = g(z, λ)-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ*(choleskyΣθ\g(z, λ)) + Hθ*βhat 

    #compute βhat_prime_lambda
    dgλz = dgλ(z, λ) 
    gλz = g(z, λ) 
    βhat_prime_lambda = choleskyXΣX\(X'*(choleskyΣθ\dgλz))

    #compute qtilde_prime_lambda
    AA = dgλz - X*βhat_prime_lambda
    BB = gλz - X*βhat 
    qtilde_prime_lambda = 2*AA'*(choleskyΣθ\BB)  

    #compute m_prime_lambda
    m_prime_lambda = Bθ*(choleskyΣθ\dgλz)+Hθ*βhat_prime_lambda

    #compute main expression 
    jac = z0 -> abs(reduce(*, map(x -> dg(x, λ), z0)))
    qC = qtilde*Cθ 
    detqC = det(qC) 
    expr = z0 -> (g(z0, λ) - m)

    bilinearform = z0 -> 1 + expr(z0)'*(qC\(expr(z0)))

    #compute derivative of Jacobian
    function djac(z0)
        dgλx_vec = z0 -> map(zi -> dgλx(zi, λ), z0)
        dg_vec = z0 -> map(zi -> dg(zi, λ), z0)
        local prod = z0 -> reduce(*, dg_vec(z0))
        local p = prod(z0) 
        p != 0 ? (AAA = dgλx_vec(z0);
        BBB = dg_vec(z0);
        ff = i -> p * AAA[i]/BBB[i] ;
        ee = reduce(+, map(ff, collect(1:length(z0))));
        dg(2, 2) < 0 ? (-1)^(length(z0))*ee : ee) : 0
    end

    #return jac, djac

    #compute derivative of main expression
    EXPR1 = z0 -> jac(z0)
    EXPR2 = z0 -> detqC^(-1/2)
    EXPR3 = z0 -> bilinearform(z0)^(-(n-p+k)/2)
    dEXPR1 = z0 -> djac(z0)
    dEXPR2 = z0 -> -0.5 * detqC^(-1/2)*tr(qC\(qtilde_prime_lambda*Cθ))
    
    qCinv = qC\I
    qC_inv_prime_lambda = - qC \ ((qtilde_prime_lambda*Cθ) * qCinv)
    
    dbilinearform = z0 -> (local AA = dgλ(z0, λ) - m_prime_lambda;
                    local BB = (g(z0, λ) - m);
                    2*AA'*(qC \ BB) + BB' * qC_inv_prime_lambda * BB
                    )
    dEXPR3 = z0 -> -(n-p+k)/2 * bilinearform(z0)^(-(n-p+k+2)/2) * dbilinearform(z0)

    #product rule
    main = z0 -> (EXPR1(z0) * EXPR2(z0) * EXPR3(z0))
    main_deriv = z0 -> (dEXPR1(z0)*EXPR2(z0)*EXPR3(z0) + dEXPR2(z0)*EXPR1(z0)*EXPR3(z0)+dEXPR3(z0)*EXPR1(z0)*EXPR2(z0))
    #sub-functions (dependence on z0)
    #return (EXPR1, dEXPR1)
    #return (EXPR2, dEXPR2)
    #return (EXPR3, dEXPR3)
    return (main, main_deriv)
    
    #sub-constants
    #return (qtilde, qtilde_prime_lambda) 
    #return (βhat, βhat_prime_lambda)
    #return (gλz, dgλz)
    #return (qCinv, qC_inv_prime_lambda)
    #return (m, m_prime_lambda)
end

"""
Compute derivative of p(theta, lambda| z) with respect to theta
"""
function posterior_theta(θ, λ, pθ, dpθ, pλ, setting)
    s = setting.s 
    s0 = setting.s0 
    X = setting.X
    X0 = setting.X0 
    z = setting.z
    g = boxCox #boxCox by default
    dg = boxCoxPrime
    n = size(X, 1) 
    p = size(X, 2) 
    k = size(X0, 1) 
    Eθ = K(s0, s0, θ, rbf) 
    Σθ = K(s, s, θ, rbf) 
    Bθ = K(s0, s, θ, rbf) 
    choleskyΣθ = cholesky(Σθ) 
    choleskyXΣX = cholesky(Hermitian(X'*(choleskyΣθ\X))) 
    Dθ = Eθ - Bθ*(choleskyΣθ\Bθ') 
    Hθ = X0 - Bθ*(choleskyΣθ\X) 
    Cθ = Dθ + Hθ*(choleskyXΣX\Hθ') 
    βhat = (X'*(choleskyΣθ\X))\(X'*(choleskyΣθ\g(z, λ))) 
    qtilde = (expr = g(z, λ)-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ*(choleskyΣθ\g(z, λ)) + Hθ*βhat 

    Σθ_inv_X = Σθ\X #precomputation 
    Σθ_prime = K(s, s, θ, rbf_prime) 
    Σθ_inv_X = Σθ\X #precomputation 

    #compute betahat_prime_theta
    expr_mid = X'*(Σθ\Σθ_prime)*(Σθ_inv_X)
    AA = choleskyXΣX\(expr_mid)*(choleskyXΣX\(X'*(Σθ\g(z, λ))))
    BB = - (choleskyXΣX\(X'*(Σθ\(Σθ_prime*(Σθ\g(z, λ))))))
    βhat_prime_theta = AA + BB
    
    #compute qtilde_prime_theta
    vv = g(z, λ) - X*βhat
    rr = X*βhat_prime_theta
    AA = (-rr)' * (Σθ \ vv)
    BB = - vv' * (Σθ \ (Σθ_prime * (Σθ \ vv)))
    CC =  vv' * (Σθ \ (-rr))
    qtilde_prime_theta = AA + BB + CC

    jac = y -> (abs(reduce( *, map(x -> dg(x, λ), y))))
    jacz = jac(z)

    EXPR1 = det(choleskyΣθ)^(-1/2)
    EXPR2 = det(choleskyXΣX)^(-1/2)
    EXPR3 = qtilde^(-(n-p)/2) 
    EXPR4 = pθ(θ)*pλ(λ)*jacz^(1-p/n)
    dEXPR1 = -0.5 * det(choleskyΣθ)^(-1/2)*tr(Σθ\Σθ_prime)
    dEXPR2 = -0.5*det(choleskyXΣX)^(-1/2)*tr(choleskyXΣX\(- Σθ_inv_X' * Σθ_prime * Σθ_inv_X))
    dEXPR3 = -((n-p)/2)*qtilde^(-(n-p+2)/2)*qtilde_prime_theta
    dEXPR4 = dpθ(θ)*pλ(λ)*jacz^(1-p/n)

    #main = EXPR1*EXPR2*EXPR3*EXPR4
    #dmain = dEXPR1 * EXPR2 * EXPR3 * EXPR4 + dEXPR2 * EXPR1 * EXPR3 * EXPR4 
    #        + dEXPR3 * EXPR1 * EXPR2 * EXPR4 + dEXPR4 * EXPR1 * EXPR2 * EXPR3 
    main = EXPR1*EXPR2*EXPR3*EXPR4
    dmain = dEXPR1 * EXPR2 * EXPR3 * EXPR4 + dEXPR2*EXPR1*EXPR3*EXPR4 + dEXPR3*EXPR1*EXPR2*EXPR4 + dEXPR4*EXPR1*EXPR2*EXPR3

    return (main, dmain)
end
  
"""
Use Taylor's Theorem with Remainder to check  
validity of computed derivative. More specifically, check 
that error of linear approximation decays like O(h^2)
"""
function checkDerivative(f, df, x0)
    f0 = f(x0)
    df0 = df(x0) 
    dx = rand(size(x0, 1), size(x0, 2))
    h = zeros(10)
    for i=1:length(h)
        h[i] = 2. ^(-i-8) 
    end
    A = zeros(length(h))
    for i = 1:length(h) 
        fi = f(x0 + h[i]*dx)
        A[i] = norm(fi .- f0 .- df0 * (h[i] * dx))
    end
    #println(A)
    #println(h)
    display(plot(log.(h), log.(A)))
    return (log.(h), log.(A))
end

end
