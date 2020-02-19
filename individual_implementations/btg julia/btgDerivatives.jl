#module btgDeriv

include("kernel.jl")
include("transforms.jl")
using Distributions
using Printf
using SpecialFunctions
using Plots
using Polynomials
using LinearAlgebra

#export prob, partial_theta, partial_lambda, partial_z0, posterior_theta, posterior_lambda, checkDerivative

"""
Precompute theta-dependent quantities and assign variable to contents of setting 
"""
function func(θ, setting)
    s = setting.s 
    s0 = setting.s0 
    X = setting.X
    X0 = setting.X0 
    z = setting.z
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
    Eθ_prime = K(s0, s0, θ, rbf_prime)
    Eθ_prime2 = K(s0, s0, θ, rbf_prime2)  
    Σθ_prime = K(s, s, θ, rbf_prime) 
    Σθ_prime2 = K(s, s, θ, rbf_prime2) 
    Bθ_prime = K(s0, s, θ, rbf_prime) 
    Bθ_prime2 = K(s0, s, θ, rbf_prime2) 
    return (s, s0, X, X0, z, n, p, k, Eθ, Σθ, Bθ, choleskyΣθ, choleskyXΣX, Dθ, Hθ, Cθ, Eθ_prime,Eθ_prime2, Σθ_prime, Σθ_prime2, Bθ_prime, Bθ_prime2)
end

"""
Compute derivative of beta hat with respect to theta
"""
function compute_betahat_prime_theta(choleskyΣθ, choleskyXΣX, expr_mid, Σθ_prime, X, gλz, Σθ_inv_X)
    AA = choleskyXΣX\(expr_mid)*(choleskyXΣX\(X'*(choleskyΣθ\gλz)))
    BB = - (choleskyXΣX\(X'*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\gλz)))))
    βhat_prime_theta = AA + BB
    βhat_prime_theta = reshape(βhat_prime_theta, size(βhat_prime_theta, 1), size(βhat_prime_theta, 2)) #turn 1D array into 2D array
end

"""
Compute second derivative of beta hat
Q: - derivative of Sigma_theta^-1 (function handle)
dQ: derivative of Q (function handle)
gλz: g_lambda(z)
"""
function compute_βhat_prime2_theta(choleskyXΣX, choleskyΣθ, expr_mid, X, Q, dQ, gλz)
    XQX = X'*Q(X) #precompute
    Qgλz = Q(gλz) #precompute
    βhatEPXR1 = - (choleskyXΣX\(-expr_mid * (choleskyXΣX\((XQX * (choleskyXΣX\(X'*(choleskyΣθ\gλz))))))))
    βhatEPXR2 = choleskyXΣX\((X'*dQ(X*(choleskyXΣX\(X'*(choleskyΣθ\gλz))))))
    βhatEPXR3  = - (choleskyXΣX\((XQX*(choleskyXΣX\(-expr_mid * (choleskyXΣX\(X'*(choleskyΣθ\(gλz)))))))))
    βhatEPXR4 = - (choleskyXΣX\(XQX*(choleskyXΣX\(X'*Qgλz))))
    βhatEPXR5 = (choleskyXΣX\(-expr_mid * (choleskyXΣX\(X'*Qgλz))))
    βhatEPXR6 = - (choleskyXΣX\(X'*dQ(gλz)))
    βhat_prime2_theta = βhatEPXR1 + βhatEPXR2 + βhatEPXR3 + βhatEPXR4 + βhatEPXR5 + βhatEPXR6
end 

"""
First derivative of qtilde with respect to theta
"""
function compute_qtilde_prime_theta(gλz, X, βhat, βhat_prime_theta, choleskyΣθ, Σθ_prime)
    meanvv = gλz - X*βhat
    rr = X*βhat_prime_theta
    AA = (-rr)' * (choleskyΣθ \ meanvv)
    BB = - meanvv' * (choleskyΣθ \ (Σθ_prime * (choleskyΣθ \ meanvv)))
    CC =  meanvv' * (choleskyΣθ \ (-rr))
    qtilde_prime_theta = AA .+ BB .+ CC
end

"""
Second derivative of qtilde with respect to theta
"""
function compute_qtilde_prime2_theta(choleskyΣθ, X, meanvv, Q, dQ, βhat_prime_theta, βhat_prime2_theta)
    Xdβ = X*βhat_prime_theta
    qtildeEXPR1 = -meanvv'*dQ(meanvv)
    qtildeEXPR2 = 2*(X*βhat_prime_theta)'*Q(meanvv) 
    qtildeEXPR3 = 2*Xdβ'*(choleskyΣθ\(Xdβ))
    qtildeEXPR4 = 2*meanvv'*Q(Xdβ)
    qtildeEXPR5 = -2*meanvv'*(choleskyΣθ\(X*βhat_prime2_theta))
    qtilde_prime2_theta = qtildeEXPR1 .+ qtildeEXPR2 .+ qtildeEXPR3 .+ qtildeEXPR4 .+ qtildeEXPR5
end

"""
First derivative of Htheta with respect to theta
"""
function compute_Hθ_prime(Bθ_prime, Σθ_inv_X, choleskyΣθ, Σθ_prime, Bθ, X)
    #compute Hθ_prime 
    AA = -Bθ_prime*Σθ_inv_X 
    #BB = Bθ*Σθ\(Σθ_prime*(Σθ\X)) displaying the bug
    BB = Bθ*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\X)))
    Hθ_prime = AA + BB
end

"""
Second derivative of Htheta with respect to theta
"""
function compute_Hθ_prime2(Bθ, Bθ_prime, Bθ_prime2, Σθ_inv_X, choleskyΣθ, X, Q, dQ)
    AA = - Bθ_prime2*(choleskyΣθ\X)
    BB = 2*Bθ_prime*Q(X)
    CC = Bθ*dQ(X)
    AA+BB+CC
end

"""
First derivative of m_theta with respect to theta
"""
function compute_m_prime_theta(Bθ, Bθ_prime, choleskyΣθ,Σθ_prime, gλz, βhat, βhat_prime_theta,Hθ, Hθ_prime)
    AA = Bθ_prime*(choleskyΣθ\gλz)
    BB = - Bθ*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\gλz)))
    CC = Hθ_prime*βhat
    DD = Hθ*βhat_prime_theta
    m_prime_theta = AA + BB + CC + DD
end

"""
First derivative of m_theta with respect to theta
"""
function compute_m_prime2_theta(Bθ, Bθ_prime, Bθ_prime2, choleskyΣθ, Σθ_prime, gλz, βhat, βhat_prime_theta, 
    βhat_prime_theta2, Hθ, Hθ_prime, Hθ_prime2, Q, dQ)
    AA = Bθ_prime2*(choleskyΣθ\gλz) - 2*Bθ_prime*(Q(gλz))
    BB = -Bθ*(dQ(gλz))
    CC = Hθ_prime2*βhat+2*Hθ_prime*βhat_prime_theta 
    DD = Hθ*βhat_prime_theta2

    m_prime_theta2 = AA + BB + CC + DD
end

"""
First derivative of D_theta with respect to theta
"""
function compute_Dθ_prime(choleskyΣθ, Bθ, Eθ_prime, Σθ_prime, Bθ_prime)
    sigma_inv_B = choleskyΣθ \ Bθ' #precomputation
    AA = Eθ_prime - Bθ_prime * sigma_inv_B 
    BB = sigma_inv_B' * Σθ_prime * sigma_inv_B
    CC = - sigma_inv_B' * Bθ_prime'
    Dθ_prime = AA + BB + CC
end

"""
Second derivative of D_theta with respect to theta
"""
function compute_Dθ_prime2(choleskyΣθ, Bθ, Bθ_prime, Bθ_prime2, Eθ_prime2, Q, dQ)
    AA = Eθ_prime2 - Bθ_prime2*(choleskyΣθ\Bθ') - Bθ*(choleskyΣθ\Bθ_prime2')
    BB = Bθ*dQ(Bθ') + 2*Bθ*Q(Bθ_prime') + 2*Bθ_prime*Q(Bθ')
    CC = -2*Bθ_prime*(choleskyΣθ\Bθ_prime')
    AA + BB + CC
end

"""
First derivative of C_theta with respect to theta
"""
function compute_Cθ_prime(Dθ_prime,Hθ,  Hθ_prime, choleskyXΣX, Σθ_inv_X, Σθ_prime)
    AA = Dθ_prime + Hθ_prime*(choleskyXΣX\Hθ')
    BB = Hθ*(choleskyXΣX\(Σθ_inv_X'*Σθ_prime*Σθ_inv_X))*(choleskyXΣX\Hθ')
    CC = Hθ*(choleskyXΣX\(Hθ_prime'))
    C_theta_prime = AA + BB + CC
end

"""
Second derivative of C_theta with respect to theta
"""
function compute_Cθ_prime2(Dθ_prime2, Hθ, Hθ_prime, Hθ_prime2, choleskyXΣX, dPinv, d2Pinv)
    AA = Dθ_prime2 + Hθ_prime2 * (choleskyXΣX\Hθ')
    BB = Hθ * (choleskyXΣX\Hθ_prime2') 
    CC = Hθ*(d2Pinv(Hθ'))
    DD = 2*(Hθ_prime * (choleskyXΣX\Hθ_prime') + Hθ_prime * dPinv(Hθ') + Hθ*(dPinv(Hθ_prime'))  )
    AA+BB+CC+DD
end

function test()
    #P, dP_inv, 
end


#"""
#Derivative of H_theta with respect to theta
#"""
#function compute_H

"""
p(z0| theta, lambda, z)
"""
function prob(θ, λ, setting)
    (s, s0, X, X0, z, n, p, k, Eθ, Σθ, Bθ, choleskyΣθ, choleskyXΣX, Dθ, Hθ, Cθ, Eθ_prime,Eθ_prime2, Σθ_prime, Σθ_prime2, Bθ_prime, Bθ_prime2) = func(θ, setting)
    g = boxCox #boxCox by default
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
    (s, s0, X, X0, z, n, p, k, Eθ, Σθ, Bθ, choleskyΣθ, choleskyXΣX, Dθ, Hθ, Cθ, Eθ_prime,Eθ_prime2, Σθ_prime, Σθ_prime2, Bθ_prime, Bθ_prime2) = func(θ, setting)
    g = boxCox #boxCox by default
    gλz = g(z, λ)

    βhat = (X'*(choleskyΣθ\X))\(X'*(choleskyΣθ\g(z, λ))) 
    qtilde = (expr = g(z, λ)-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ*(choleskyΣθ\g(z, λ)) + Hθ*βhat 

    Σθ_inv_X = Σθ\X #precomputation 
    tripleΣ = Σθ_inv_X' * Σθ_prime * Σθ_inv_X  #precompute X'Sigma^-1 dSigma Sigma^-1 X 
    dQ = Y -> (choleskyΣθ\((Σθ_prime2*(choleskyΣθ\Y)))) - 2*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Y))))))
    Q = Y -> (Σθ\(Σθ_prime*(Σθ\(Y))))
    dPinv = Y -> choleskyXΣX\(X'*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(X*(choleskyXΣX\Y))))))
    XQX = X'*Q(X)
    dP = Y -> -XQX*Y
    d2P = Y -> -X'*(dQ(X)*Y)
    d2Pinv = Y -> -dPinv(dP(choleskyXΣX\Y)) - (choleskyXΣX\(d2P(choleskyXΣX\Y))) - (choleskyXΣX\(dP(dPinv(Y))))

    #cc = gamma((n-p+k)/2)/gamma((n-p)/2)/pi^(k/2) #constant term
    
    expr_mid = X'*(choleskyΣθ\(Σθ_prime * Σθ_inv_X))#precompute
    #first derivatives
    βhat_prime_theta = compute_betahat_prime_theta(choleskyΣθ, choleskyXΣX, expr_mid, Σθ_prime, X, gλz, Σθ_inv_X)
    qtilde_prime_theta = compute_qtilde_prime_theta(gλz, X, βhat, βhat_prime_theta, choleskyΣθ, Σθ_prime)
    Hθ_prime = compute_Hθ_prime(Bθ_prime, Σθ_inv_X, choleskyΣθ, Σθ_prime, Bθ, X)
    m_prime_theta = compute_m_prime_theta(Bθ, Bθ_prime, choleskyΣθ,Σθ_prime, gλz, βhat, βhat_prime_theta,Hθ, Hθ_prime)
    Dθ_prime = compute_Dθ_prime(choleskyΣθ, Bθ, Eθ_prime, Σθ_prime, Bθ_prime)
    Cθ_prime = compute_Cθ_prime(Dθ_prime,Hθ,  Hθ_prime, choleskyXΣX, Σθ_inv_X, Σθ_prime)

    #second derivatives
    βhat_prime2_theta  = compute_βhat_prime2_theta(choleskyXΣX, choleskyΣθ, expr_mid, X, Q, dQ, gλz)
    Hθ_prime2 = compute_Hθ_prime2(Bθ, Bθ_prime, Bθ_prime2, Σθ_inv_X, choleskyΣθ, X, Q, dQ)
    m_prime2_theta = compute_m_prime2_theta(Bθ, Bθ_prime, Bθ_prime2, choleskyΣθ, Σθ_prime, gλz, βhat, βhat_prime_theta, 
    βhat_prime2_theta, Hθ, Hθ_prime, Hθ_prime2, Q, dQ)
    Dθ_prime2 = compute_Dθ_prime2(choleskyΣθ, Bθ, Bθ_prime, Bθ_prime2, Eθ_prime2, Q, dQ)
    Cθ_prime2 = compute_Cθ_prime2(Dθ_prime2, Hθ, Hθ_prime, Hθ_prime2, choleskyXΣX, dPinv, d2Pinv)
    
    #compute derivative of main expression
    expr = z0 -> g(z0, λ) .- m
    qC = qtilde*Cθ 
    bilinearform = z0 -> 1 .+ expr(z0)'*(qC\(expr(z0)))
    qC_inv = qC\I
    detqC = det(qC) 
    qC_prime_theta =  qtilde_prime_theta .* Cθ + qtilde .* Cθ_prime
    AA = -0.5 * detqC^(-1/2) * tr(qC\(qC_prime_theta)) 
    qC_inv_prime_theta = - qC\(qC_prime_theta * qC_inv)  

    BB = z0 -> -m_prime_theta'*(qC\(expr(z0)))
    CC = z0 -> expr(z0)' * qC_inv_prime_theta * expr(z0)
    DD = z0 -> -expr(z0)'*(qC\m_prime_theta)
    EE = z0 -> bilinearform(z0)^(-(n-p+k)/2)
    FF = z0 -> detqC^(-1/2) * (-(n-p+k)/2) * (bilinearform(z0))^(-(n-p+k+2)/2)
    
    dmain = z0 -> [(AA*EE(z0) .+ FF(z0)*(BB(z0) .+ CC(z0) .+ DD(z0)))]
    main = z0 -> (detqC^(-1/2))*(bilinearform(z0))^(-(n-p+k)/2)

    if false 
        println("Σθ")
        println(Σθ)
        println("Σθ_prime")
        println(Σθ_prime)
        println("Σθ_prime2")
        println(Σθ_prime2)
        println("Bθ_prime")
        println(Bθ_prime)
        println("Bθ_prime2")
        println(Bθ_prime2)
    end
    return (Cθ_prime, Cθ_prime2)
    #return (vec(Bθ_prime), vec(Bθ_prime2))
    #return (m_prime_theta, m_prime2_theta)
    #return (vec(Eθ_prime), vec(Eθ_prime2))
    #return (vec(Bθ), vec(Bθ_prime))
    #return (vec(Bθ_prime), vec(Bθ_prime2))
    #return (vec(Q(X)), vec(dQ(X)))
    #return (vec(Dθ), vec(Dθ_prime))
    #return (vec(Dθ_prime), vec(Dθ_prime2))
    #return (vec(dP(X')), vec(d2P(X')))
    #return (vec(dPinv(X')), vec(d2Pinv(X')))
    #main_deriv = (AA*EE(z0) + FF(z0)*(BB(z0) + CC(z0) + DD(z0)))
    #return (vec(Σθ), vec(Σθ_prime))
    #return (βhat, βhat_prime_theta)
    #return ([qtilde], [qtilde_prime_theta])
    #return (vec(Eθ), vec(Eθ_prime))
    #return (vec(Bθ), vec(Bθ_prime))
    #return (vec(Σθ), vec(Σθ_prime))
    #return (vec(Hθ), vec(Hθ_prime))
    #return (m, m_prime_theta)
    #return (vec(Dθ), vec(Dθ_prime))
    #return (vec(Cθ), vec(C_theta_prime))
    
    #return (main, dmain)
    #return (vec(qC_inv), vec(qC_inv_prime_theta))
    #return ([detqC^(-1/2)], [AA])
    #return (bilinearform, z0 -> BB(z0) + CC(z0) + DD(z0))
end

"""
Compute derivative of p(z0|theta, lambda, z) w.r.t lambda
"""
function partial_lambda(θ, λ, setting)
    (s, s0, X, X0, z, n, p, k, Eθ, Σθ, Bθ, choleskyΣθ, choleskyXΣX, Dθ, Hθ, Cθ, Eθ_prime,Eθ_prime2, Σθ_prime, Σθ_prime2, Bθ_prime, Bθ_prime2) = func(θ, setting)

    #transformation and its derivatives
    g = boxCox #boxCox by default
    dg = boxCoxPrime
    dgλ = boxCoxPrime_lambda
    dgλ2 = boxCoxPrime_lambda2
    dgλx = boxCoxMixed_lambda_z

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
    expr = z0 -> (g(z0, λ) .- m)

    bilinearform = z0 -> 1 .+ expr(z0)'*(qC\(expr(z0)))

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
Compute deriative of p(z0|theta, lambda, z) with respect to z0
"""
function partial_z0(θ, λ, setting, g = boxCox, dg = boxCoxPrime, dg2 = boxCoxPrime2)
    (s, s0, X, X0, z, n, p, k, Eθ, Σθ, Bθ, choleskyΣθ, choleskyXΣX, Dθ, Hθ, Cθ, Eθ_prime,Eθ_prime2, Σθ_prime, Σθ_prime2, Bθ_prime, Bθ_prime2) = func(θ, setting)

    βhat = choleskyXΣX\(X'*(choleskyΣθ\g(z, λ))) 
    qtilde = (expr = g(z, λ)-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ*(choleskyΣθ\g(z, λ)) + Hθ*βhat

    jac = z0 -> abs(reduce(*, map(x -> dg(x, λ), z0)))

    djac = z0 ->(local n = length(z0);
                reshape(
                (local dd = map(x->dg(x, λ), z0); 
                local p = abs(reduce(*, dd));
                (f = i -> (dd[i] !=0 ? p/dd[i]*sign(dd[i])*(-1)^(((sign(dd[mod(i+1, n)+1])+1)/2)+1)* 
                dg2(z0[i], λ) : 0) ; 
                map(f, collect(1:n)))), n, 1))

    qC = qtilde*Cθ
    detqC = det(qC)
    
    ccc = gamma((n-p+k)/2)/gamma((n-p)/2)/pi^(k/2)/detqC^(1/2) #constant 

    expr = z0 -> (g(z0, λ) .- m)'*(qC\(g(z0, λ) .- m))

    dexpr = z0 -> (local n = length(z0);
                    2 * Diagonal(dg(z0, λ)) * (qC\(g(z0, λ) .- m));
                    )

    #expr2 = z0 -> (1 .+ (g(z0, λ) .- m)'*(qC\(g(z0, λ) .- m)))^(-(n-p+k)/2)
    #dexpr2 = z0 -> (-(n-p+k)/2) * (1 .+ (g(z0, λ) .- m)'*(qC\(g(z0, λ) .- m)))^(-(n-p+k+2)/2) * 2 * Diagonal(dg(z0, λ)) * (qC\(g(z0, λ) .- m))

    expr3 = z0 -> ccc*jac(z0)*(1 .+ (g(z0, λ) .- m)'*(qC\(g(z0, λ) .- m)))^(-(n-p+k)/2)
    dexpr3 = z0 -> ccc*(jac(z0)*(-(n-p+k)/2) * (1 .+ (g(z0, λ) .- m)'*(qC\(g(z0, λ) .- m)))^(-(n-p+k+2)/2) * 2 * Diagonal(dg(z0, λ)) * (qC\(g(z0, λ) .- m)) 
                    + djac(z0)*(1 .+ (g(z0, λ) .- m)'*(qC\(g(z0, λ) .- m)))^(-(n-p+k)/2))

    #return (z0 -> g(z0, λ), z0 -> [dg(z0, λ)[1] 0 ; 0 dg(z0, λ)[2]])
    #return (expr, dexpr)
    #return (jac, djac)
    return (expr3, dexpr3)
end

"""
Compute derivative of p(theta, lambda| z) with respect to theta
"""
function posterior_theta(θ, λ, pθ, dpθ, dpθ2, pλ, setting)
    (s, s0, X, X0, z, n, p, k, Eθ, Σθ, Bθ, choleskyΣθ, choleskyXΣX, Dθ, Hθ, Cθ, Eθ_prime,Eθ_prime2, Σθ_prime, Σθ_prime2, Bθ_prime, Bθ_prime2) = func(θ, setting)
    g = boxCox
    dg = boxCoxPrime
    gλz = g(z, λ)
    βhat = (X'*(choleskyΣθ\X))\(X'*(choleskyΣθ\gλz)) 
    qtilde = (expr = gλz-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ*(choleskyΣθ\gλz) + Hθ*βhat 
    Σθ_inv_X = Σθ\X #precompute
    meanvv = gλz - X*βhat #precompute

    #compute betahat_prime_theta
    expr_mid = X'*(choleskyΣθ\Σθ_prime)*(Σθ_inv_X)
    βhat_prime_theta = compute_betahat_prime_theta(choleskyΣθ, choleskyXΣX, expr_mid, Σθ_prime, X, gλz, Σθ_inv_X)

    #compute qtilde_prime_theta
    qtilde_prime_theta  = compute_qtilde_prime_theta(gλz, X, βhat, βhat_prime_theta, choleskyΣθ, Σθ_prime)

    jac = y -> (abs(reduce( *, map(x -> dg(x, λ), y))))
    jacz = jac(z)

    EXPR1 = det(choleskyΣθ)^(-1/2)
    EXPR2 = det(choleskyXΣX)^(-1/2)
    EXPR3 = qtilde^(-(n-p)/2) 
    EXPR4 = pθ(θ)*pλ(λ)*jacz^(1-p/n)
    trΣθqΣθ_prime = tr(choleskyΣθ\Σθ_prime) #precompute 
    dEXPR1 = -0.5 * det(choleskyΣθ)^(-1/2)*trΣθqΣθ_prime
    XΣdΣΣX = Σθ_inv_X' * Σθ_prime * Σθ_inv_X  #precompute
    trexpr = tr(choleskyXΣX\(-XΣdΣΣX))#precompute
    dEXPR2 = -0.5*det(choleskyXΣX)^(-1/2)*trexpr
    dEXPR3 = -((n-p)/2)*qtilde^(-(n-p+2)/2)*qtilde_prime_theta
    dEXPR4 = dpθ(θ)*pλ(λ)*jacz^(1-p/n)

    main = EXPR1*EXPR2*EXPR3*EXPR4
    dmain = dEXPR1 * EXPR2 * EXPR3 * EXPR4 .+ dEXPR2*EXPR1*EXPR3*EXPR4 .+ dEXPR3*EXPR1*EXPR2*EXPR4 .+ dEXPR4*EXPR1*EXPR2*EXPR3

    #-====================== dmain2 (second derivative)===============================
    d2EXPR1 = 0.25 * EXPR1 * trΣθqΣθ_prime^2 -0.5 * EXPR1* tr(choleskyΣθ\Σθ_prime2 - choleskyΣθ\(Σθ_prime*(choleskyΣθ\Σθ_prime)))

    dQ = Y -> (choleskyΣθ\((Σθ_prime2*(choleskyΣθ\Y)))) - 2*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Y))))))
    dPinv = Y -> choleskyXΣX\(X'*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(X*(choleskyXΣX\Y))))))

    d2EXPR2 = 0.25 * EXPR2 * trexpr^2 .+ 0.5*EXPR2*tr(dPinv(XΣdΣΣX) .+ choleskyXΣX\(X'*dQ(X)))

    Q = Y -> (Σθ\(Σθ_prime*(Σθ\(Y))))
    βhat_prime2_theta = compute_βhat_prime2_theta(choleskyXΣX, choleskyΣθ, expr_mid, X, Q, dQ, gλz)
    #meanvv is gλz - X*βhat
    #Xdβ = X*βhat_prime_theta
    #qtildeEXPR1 = -meanvv'*dQ(meanvv)
    #qtildeEXPR2 = 2*(X*βhat_prime_theta)'*Q(meanvv) 
    #qtildeEXPR3 = 2*Xdβ'*(choleskyΣθ\(Xdβ))
    #qtildeEXPR4 = 2*meanvv'*Q(Xdβ)
    #qtildeEXPR5 = -2*meanvv'*(choleskyΣθ\(X*βhat_prime2_theta))

    #qtilde_prime2_theta = qtildeEXPR1 .+ qtildeEXPR2 .+ qtildeEXPR3 .+ qtildeEXPR4 .+ qtildeEXPR5
    qtilde_prime2_theta = compute_qtilde_prime2_theta(choleskyΣθ, X, meanvv, Q, dQ, βhat_prime_theta, βhat_prime2_theta)
    
    #dEXPR3 =  -((n-p)/2)*qtilde^(-(n-p+2)/2)*qtilde_prime_theta
    d2EXPR3 = ((n-p)/2)*((n-p+2)/2)*qtilde^(-(n-p+4)/2)*qtilde_prime_theta^2 - (n-p)/2 * qtilde^(-(n-p+2)/2)*qtilde_prime2_theta
    d2EXPR4 = dpθ2(θ)*pλ(λ)*jacz^(1-p/n)
    
    d2main = (d2EXPR1*EXPR2*EXPR3*EXPR4 .+ d2EXPR2*EXPR1*EXPR3*EXPR4 .+ d2EXPR3*EXPR1*EXPR2*EXPR4 .+ d2EXPR4*EXPR1*EXPR2*EXPR3 
                .+ 2*(dEXPR1*dEXPR2*EXPR3*EXPR4 .+ dEXPR1*EXPR2*dEXPR3*EXPR4 .+ dEXPR1*EXPR2*EXPR3*dEXPR4 .+ EXPR1*dEXPR2*dEXPR3*EXPR4
                .+ EXPR1*dEXPR2*EXPR3*dEXPR4 .+ EXPR1*EXPR2*dEXPR3*dEXPR4))
                if true
                    println("===============================================================")
                    println("theta");println(θ);println("betahat");println(βhat);println("glambdaz");println(gλz);println("qtilde")
                    println(qtilde);println("m");println(m);println("expr_mid");println(expr_mid);println("βhat_prime2_theta")
                    println(βhat_prime2_theta);println(" qtilde_prime2_theta");println(qtilde_prime2_theta)
                    println("jacz");println(jacz);println("jacz^(1-p/n)");println(jacz^(1-p/n));println("pθ(θ)")
                    println(pθ(θ)); println("pλ(λ)");println(pλ(λ));println("EXPR1");println(EXPR1);println("EXPR2");println(EXPR2)
                    println("EXPR3"); println(EXPR3);println("EXPR4");println(EXPR4);println("dEXPR1"); println(dEXPR1)
                    println("dEXPR2"); println(dEXPR2);println("dEXPR3");println(dEXPR3);println("dEXPR4")
                    println(dEXPR4); println("d2EXPR1");println(d2EXPR1);println("d2EXPR2");println(d2EXPR2)
                    println("d2EXPR3");println(d2EXPR3);println("d2EXPR4");println(d2EXPR4);println("main")
                    println(main);println("dmain");println(dmain);println("d2main");println(d2main)
                    println("===============================================================")
                end

    return (dmain, d2main)
    #return (qtilde_prime_theta, qtilde_prime2_theta)
    #return (choleskyXΣX\(expr_mid)*(choleskyXΣX\(X'*(Σθ\gλz))), βhatEPXR1+βhatEPXR2+βhatEPXR3+βhatEPXR4)
    #return (- (choleskyXΣX\(X'*(Σθ\(Σθ_prime*(Σθ\gλz))))), βhatEPXR5 + βhatEPXR6)
    #return (Q(gλz), dQ(gλz))
    #return (βhat_prime_theta, βhat_prime2_theta)
    #return (vec(choleskyXΣX\X), vec(dPinv(X)))
    #return (Σθ\(Σθ_prime*(Σθ\X)), dQ(X))
    #return (dEXPR2, d2EXPR2)
    #return (main, dmain)

end
  
"""
Compute derivative of p(theta, lambda| z) with respect to theta
"""
function posterior_lambda(θ, λ, pθ, pλ, dpλ, setting)
    (s, s0, X, X0, z, n, p, k, Eθ, Σθ, Bθ, choleskyΣθ, choleskyXΣX, Dθ, Hθ, Cθ, Eθ_prime,Eθ_prime2, Σθ_prime, Σθ_prime2, Bθ_prime, Bθ_prime2) = func(θ, setting)

    g = boxCox
    dg = boxCoxPrime
    dgλ = boxCoxPrime_lambda
    dgλx = boxCoxMixed_lambda_z

    βhat = (X'*(choleskyΣθ\X))\(X'*(choleskyΣθ\g(z, λ))) 
    qtilde = (expr = g(z, λ)-X*βhat; expr'*(choleskyΣθ\expr)) 

    #compute βhat_prime_lambda
    dgλz = dgλ(z, λ) 
    gλz = g(z, λ) 
    βhat_prime_lambda = choleskyXΣX\(X'*(choleskyΣθ\dgλz))

    #compute qtilde_prime_lambda
    AA = dgλz - X*βhat_prime_lambda
    BB = gλz - X*βhat 
    qtilde_prime_lambda = 2*AA'*(choleskyΣθ\BB) 

    jac = z0 -> (abs(reduce( *, map(x -> dg(x, λ), z0))))
    jacz = jac(z)

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

    ccc = det(choleskyΣθ)^(-1/2)*det(choleskyXΣX)^(-1/2)*pθ(θ)
    EXPR1 = qtilde^(-(n-p)/2)
    EXPR2 = jacz^(1-p/n)
    EXPR3 = pλ(λ)
    dEXPR1 = -(n-p)/2*qtilde^(-(n-p+2)/2)*qtilde_prime_lambda
    dEXPR2 = (1-p/n)*jacz^(-p/n)*djac(z)
    dEXPR3 = dpλ(λ)
    main = ccc * EXPR1*EXPR2*EXPR3 
    dmain = ccc*(dEXPR1*EXPR2*EXPR3 + dEXPR2*EXPR1*EXPR3 + dEXPR3*EXPR1*EXPR2)
    return (main, dmain)
end
    
"""
Use Taylor's Theorem with Remainder to check  
validity of computed derivative. More specifically, check 
that error of linear approximation decays like O(h^2)

last: smallest power of exp^(-x) we wish to compute in scheme
"""
function checkDerivative(f, df, x0, first = 3, last = 12, num = 10)
    f0 = f(x0)
    df0 = df(x0) 
    if size(x0, 2)>1
        dx = rand(size(x0, 1), size(x0, 2))
    else
        dx = rand(size(x0, 1))
    end
    h = collect(first:(last-first)/num:last)
    for i=1:length(h)
        h[i] = exp(-h[i]) 
    end
    A = zeros(length(h))
    for i = 1:length(h) 
        #println(x0)
        #println(h[i]*dx)
        fi = f(x0 .+ h[i]*dx)

        if false #debug
            println("dx: ", dx)
            println("fi: ", fi)
            println("f0: ", f0)
            println("df0: ", df0)
            println("increment", h[i]*dx)
        end
        try 
            A[i] = norm((fi .- f0) .- df0' * (h[i] * dx))
        catch DimensionMismatch #we catch the case when f: R^1 -> R^n, in which case  df0'*dx will yield an error
            #println("caught in check deriv")
            A[i] = norm((fi .- f0) .- df0 .* (h[i] * dx))
        end
    end
    #println(A)
    #println(h)

    return (log.(h), log.(A))
end

