#TODO
# compute partial derivatives by chain-ruling built-in T-Dist pdf and cdf with derivatives of sub-functions 

include("../kernels/kernel.jl")
include("../transforms.jl")
include("buffers.jl")
include("../priors.jl")
include("../computation/buffers.jl")

using Distributions
using SpecialFunctions
using Plots
using Polynomials
using LinearAlgebra

#export prob, partial_theta, partial_lambda, partial_z0, posterior_theta, posterior_lambda, checkDerivative

"""
Exemplifies how slow regular K can be (when for loop are used)
"""
function time_func_artifact(θ::Float64, setting::setting{Array{Float64, 2}, Array{Float64, 1}})
    s = setting.s
    s0 = setting.s0
    X = setting.X
    X0 = setting.X0
    z = setting.z
    reset_timer!()
    @timeit "Eθ" Eθ = K(s0, s0, θ, rbf)
    @timeit "Σθ" Σθ =  K(s, s, θ, rbf)
    @timeit "fasterΣθ" Σθ =  fastK(s, s, θ, rbf_single)
    @timeit "Bθ" Bθ =  K(s0, s, θ, rbf)
    @timeit "choleskyΣθ" choleskyΣθ = cholesky(Σθ) 
    @timeit "choleskyXΣX" choleskyXΣX = cholesky(Hermitian(X'*(choleskyΣθ\X))) 
    @timeit "Dθ" Dθ = Eθ - Bθ*(choleskyΣθ\Bθ') 
    @timeit "Hθ" Hθ = X0 - Bθ*(choleskyΣθ\X) 
    @timeit "Cθ" Cθ = Dθ + Hθ*(choleskyXΣX\Hθ') 
    @timeit "Eθ_prime" Eθ_prime = K(s0, s0, θ, rbf_prime)
    @timeit "Eθ_prime2" Eθ_prime2 = K(s0, s0, θ, rbf_prime2)  
    @timeit "Σθ_prime" Σθ_prime = K(s, s, θ, rbf_prime) 
    @timeit "Σθ_prime_faster" Σθ_prime = fastK(s, s, θ, rbf_prime_single) 
    @timeit "Σθ_prime2" Σθ_prime2 = K(s, s, θ, rbf_prime2) 
    @timeit "Σθ_prime2_faster" Σθ_prime2 = fastK(s, s, θ, rbf_prime2_single) 
    @timeit "Bθ_prime" Bθ_prime = K(s0, s, θ, rbf_prime) 
    @timeit "Bθ_prime2" Bθ_prime2 = K(s0, s, θ, rbf_prime2) 
    #print_timer()
    return θ_params(Eθ, Σθ, Bθ, Dθ, Hθ, Cθ, Eθ_prime, Eθ_prime2, Σθ_prime, Σθ_prime2, Bθ_prime, Bθ_prime2, choleskyΣθ, choleskyXΣX)
end

"""
Compute derivative of beta hat with respect to theta
"""
function compute_betahat_prime_theta(choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, choleskyXΣX::Cholesky{Float64,Array{Float64, 2}}, expr_mid, Σθ_prime, X::Array{Float64,2}, gλz, Σθ_inv_X)
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
function compute_βhat_prime2_theta(choleskyXΣX::Cholesky{Float64,Array{Float64, 2}}, choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, expr_mid, X::Array{Float64,2}, Q, dQ, gλz)
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
function compute_qtilde_prime_theta(gλz, X::Array{Float64,2}, βhat, βhat_prime_theta, choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, Σθ_prime::Array{Float64,2})
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
function compute_qtilde_prime2_theta(choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, X::Array{Float64,2}, meanvv, Q, dQ, βhat_prime_theta, βhat_prime2_theta)
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
function compute_Hθ_prime(Bθ_prime::Array{Float64,2}, Σθ_inv_X::Array{Float64,2}, choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, Σθ_prime::Array{Float64,2}, Bθ::Array{Float64,2}, X::Array{Float64,2})
    #compute Hθ_prime 
    AA = -Bθ_prime*Σθ_inv_X 
    #BB = Bθ*Σθ\(Σθ_prime*(Σθ\X)) displaying the bug
    BB = Bθ*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\X)))
    Hθ_prime = AA + BB
end

"""
Second derivative of Htheta with respect to theta
"""
function compute_Hθ_prime2(Bθ::Array{Float64,2}, Bθ_prime::Array{Float64,2}, Bθ_prime2::Array{Float64,2}, Σθ_inv_X, choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, X, Q, dQ)
    AA = - Bθ_prime2*(choleskyΣθ\X)
    BB = 2*Bθ_prime*Q(X)
    CC = Bθ*dQ(X)
    AA+BB+CC
end

"""
First derivative of m_theta with respect to theta
"""
function compute_m_prime_theta(Bθ::Array{Float64,2}, Bθ_prime::Array{Float64,2}, choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, Σθ_prime::Array{Float64,2}, gλz, βhat, βhat_prime_theta,Hθ::Array{Float64,2}, Hθ_prime::Array{Float64,2})
    AA = Bθ_prime*(choleskyΣθ\gλz)
    BB = - Bθ*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\gλz)))
    CC = Hθ_prime*βhat
    DD = Hθ*βhat_prime_theta
    m_prime_theta = AA + BB + CC + DD
end

"""
First derivative of m_theta with respect to theta
"""
function compute_m_prime2_theta(Bθ::Array{Float64,2}, Bθ_prime::Array{Float64,2}, Bθ_prime2::Array{Float64,2}, choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, Σθ_prime::Array{Float64,2}, gλz, βhat, βhat_prime_theta, 
    βhat_prime2_theta, Hθ, Hθ_prime, Hθ_prime2, Q, dQ)
    AA = Bθ_prime2*(choleskyΣθ\gλz) - 2*Bθ_prime*(Q(gλz))
    BB = -Bθ*(dQ(gλz))
    CC = Hθ_prime2*βhat+2*Hθ_prime*βhat_prime_theta 
    DD = Hθ*βhat_prime2_theta

    m_prime_theta2 = AA + BB + CC + DD
end

"""
First derivative of D_theta with respect to theta
"""
function compute_Dθ_prime(choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, Bθ::Array{Float64,2}, Eθ_prime::Array{Float64,2}, Σθ_prime::Array{Float64,2}, Bθ_prime::Array{Float64,2})
    sigma_inv_B = choleskyΣθ \ Bθ' #precomputation
    AA = Eθ_prime - Bθ_prime * sigma_inv_B 
    BB = sigma_inv_B' * Σθ_prime * sigma_inv_B
    CC = - sigma_inv_B' * Bθ_prime'
    Dθ_prime = AA + BB + CC
end

"""
Second derivative of D_theta with respect to theta
"""
function compute_Dθ_prime2(choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, Bθ::Array{Float64,2}, Bθ_prime::Array{Float64,2}, Bθ_prime2::Array{Float64,2}, Eθ_prime2::Array{Float64,2}, Q, dQ)
    AA = Eθ_prime2 - Bθ_prime2*(choleskyΣθ\Bθ') - Bθ*(choleskyΣθ\Bθ_prime2')
    BB = Bθ*dQ(Bθ') + 2*Bθ*Q(Bθ_prime') + 2*Bθ_prime*Q(Bθ')
    CC = -2*Bθ_prime*(choleskyΣθ\Bθ_prime')
    AA + BB + CC
end

"""
First derivative of C_theta with respect to theta
"""
function compute_Cθ_prime(Dθ_prime::Array{Float64,2}, Hθ::Array{Float64,2},  Hθ_prime::Array{Float64,2}, choleskyXΣX::Cholesky{Float64,Array{Float64, 2}}, Σθ_inv_X::Array{Float64,2}, Σθ_prime::Array{Float64,2})
    AA = Dθ_prime + Hθ_prime*(choleskyXΣX\Hθ')
    BB = Hθ*(choleskyXΣX\(Σθ_inv_X'*Σθ_prime*Σθ_inv_X))*(choleskyXΣX\Hθ')
    CC = Hθ*(choleskyXΣX\(Hθ_prime'))
    C_theta_prime = AA + BB + CC
end

"""
Second derivative of C_theta with respect to theta
"""
function compute_Cθ_prime2(Dθ_prime2::Array{Float64,2}, Hθ::Array{Float64,2}, Hθ_prime::Array{Float64,2}, Hθ_prime2::Array{Float64,2}, choleskyXΣX::Cholesky{Float64,Array{Float64, 2}}, dPinv, d2Pinv)
    AA = Dθ_prime2 + Hθ_prime2 * (choleskyXΣX\Hθ')
    BB = Hθ * (choleskyXΣX\Hθ_prime2') 
    CC = Hθ*(d2Pinv(Hθ'))
    DD = 2*(Hθ_prime * (choleskyXΣX\Hθ_prime') + Hθ_prime * dPinv(Hθ') + Hθ*(dPinv(Hθ_prime'))  )
    AA+BB+CC+DD
end


"""
p(z0| theta, lambda, z). Computes T-Distribution explicitly (and unstably). 
"""
function prob_artifact(θ, λ, setting, type = "Gaussian")
    s = setting.s; s0 = setting.s0; X = setting.X; X0 = setting.X0; z = setting.z; n = size(X, 1); p = size(X, 2); k = size(X0, 1)  #unpack setting
    theta_params = funcθ(θ, train, test, type)
    g = boxCox #boxCox by default
    Eθ = theta_params.Eθ
    Σθ = theta_params.Σθ
    Bθ = theta_params.Bθ
    Dθ = theta_params.Dθ
    Hθ = theta_params.Hθ
    Cθ = theta_params.Cθ
    Σθ_inv_X = theta_params.Σθ_inv_X
    choleskyΣθ = theta_params.choleskyΣθ
    choleskyXΣX = theta_params.choleskyXΣX

    βhat = (X'*(choleskyΣθ\X))\(X'*(choleskyΣθ\g(z, λ)))
    qtilde = (expr = g(z, λ)-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ*(choleskyΣθ\g(z, λ)) + Hθ*βhat  
    #t = LocationScale(m[1], sqrt(qtilde[1]*Cθ[1]/(n-p)), TDist(n-p))
    #p = z0 -> Distributions.pdf(t, g(z0, λ))
    #return p
    cc = gamma((n-p+k)/2)/gamma((n-p)/2)/pi^(k/2) #constant term
    expr = z0 -> g(z0, λ) .- m
    return z0 -> cc*(det(qtilde*Cθ)^(-1/2))*(1+expr(z0)'*((qtilde*Cθ)\expr(z0)))^(-(n-p+k)/2)
end


"""
Compute derivative of p(z0|theta, lambda, z) w.r.t theta. This function should compute things that depend on BOTH theta and lambda. It
takes in all pertinent theta-dependent quantities as inputs.

WARNING: not tested, because want to get rid of cc (the unstably computed constant term), and redo some computations to use built-in cdf and pdf
"""
function partial_theta(θ::Float64, λ::Float64, train::trainingData{A, B}, test::testingData{A}, transforms, theta_params::Union{θ_params{A}, 
    Cholesky{Float64, A}, θ_param_derivs{A, Cholesky{Float64, A}}, Nothing} = nothing, type::String = "Gaussian") where A<:Array{Float64, 2} where B<:Array{Float64, 1}
    if theta_params === nothing
        #println("WARNING: recompute theta_params in partial_theta")
        theta_params = funcθ(θ, train, test, type)
        #(Eθ, Σθ, Bθ, Dθ, Hθ, Cθ, Eθ_prime, Eθ_prime2, Σθ_prime, Σθ_prime2, Bθ_prime, Bθ_prime2, choleskyΣθ, choleskyXΣX) = getθ_Params(theta_params)
        #(s, s0, X, X0, z,  n, p, k) = getSettingParams(setting)
    end
    if type == "Gaussian" || type == "Turan"
        Eθ = theta_params.Eθ
        Σθ = theta_params.Σθ
        Bθ = theta_params.Bθ
        Dθ = theta_params.Dθ
        Hθ = theta_params.Hθ
        Cθ = theta_params.Cθ
        Σθ_inv_X = theta_params.Σθ_inv_X
        choleskyΣθ = theta_params.choleskyΣθ
        choleskyXΣX = theta_params.choleskyXΣX
    else
        throw(ArgumentError("Quadrature type undefined. Please enter \"Gaussian\" or \"Turan\" for last arg."))
    end
    if type == "Turan" #load higher derivatives
        Eθ_prime = theta_params.Eθ_prime
        Eθ_prime2 = theta_params.Eθ_prime2
        Σθ_prime = theta_params.Σθ_prime
        Σθ_prime2 = theta_params.Σθ_prime2
        Bθ_prime = theta_params.Bθ_prime
        Bθ_prime2 = theta_params.Bθ_prime2
        Dθ_prime = theta_params.Dθ_prime
        Dθ_prime2 = theta_params.Dθ_prime2
        Hθ_prime = theta_params.Hθ_prime
        Hθ_prime2 = theta_params.Hθ_prime2
        Cθ_prime = theta_params.Cθ_prime
        Cθ_prime2 = theta_params.Cθ_prime2
        tripleΣ = theta_params.tripleΣ
    end

    s = train.s; s0 = test.s0; X = train.X; X0 = test.X0; z = train.z; n = size(X, 1); p = size(X, 2); k = size(X0, 1)  #unpack 

    g = boxCox #boxCox by default
    gλz = g(z, λ)

    @timeit "βhat" βhat = (X'*(choleskyΣθ\X))\(X'*(choleskyΣθ\g(z, λ))) 
    @timeit "qtilde" qtilde = (expr = g(z, λ)-X*βhat; expr'*(choleskyΣθ\expr)) 
    @timeit "m" m = Bθ*(choleskyΣθ\g(z, λ)) + Hθ*βhat 
    meanvv = gλz - X*βhat

    dg = transforms.df
    jac = z0 -> abs(reduce(*, map(x -> dg(x, λ), z0)))

    cc = gamma((n-p+k)/2)/gamma((n-p)/2)/pi^(k/2) #constant term
    t = LocationScale(m[1], sqrt(qtilde[1]*Cθ[1]/(n-p)), TDist(n-p))
    vanillat = LocationScale(0, 1, TDist(n-p))
    vanillaT_pdf = z0 -> Distributions.pdf.(vanillat, z0)
    T_pdf = z0 -> Distributions.pdf.(t, g(z0, λ)) #no Jacobian
    
    main_pdf = z0 -> Distributions.pdf.(t, g(z0, λ)) * jac(z0)
    main_cdf = z0 -> Distributions.cdf.(t, g(z0, λ)) 
    #This block of code computes first and second derivatives. 
    if type == "Turan"
        @timeit "expr_mid" expr_mid = X'*(choleskyΣθ\(Σθ_prime * Σθ_inv_X))#precompute
        #first derivatives
        @timeit "some first derivs, which depend on θ AND λ" begin
        βhat_prime_theta = compute_betahat_prime_theta(choleskyΣθ, choleskyXΣX, expr_mid, Σθ_prime, X, gλz, Σθ_inv_X)
        qtilde_prime_theta = compute_qtilde_prime_theta(gλz, X, βhat, βhat_prime_theta, choleskyΣθ, Σθ_prime)
        m_prime_theta = compute_m_prime_theta(Bθ, Bθ_prime, choleskyΣθ,Σθ_prime, gλz, βhat, βhat_prime_theta,Hθ, Hθ_prime)
        end
        #copied from func(theta, examples) in structures.jl
        dQ = Y -> (choleskyΣθ\((Σθ_prime2*(choleskyΣθ\Y)))) - 2*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Y))))))
        Q = Y -> (choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Y))))
        #second derivatives
        #@timeit "second derivs" begin
        @timeit "βhat_prime2_theta" βhat_prime2_theta  = compute_βhat_prime2_theta(choleskyXΣX, choleskyΣθ, expr_mid, X, Q, dQ, gλz)
        @timeit "qtilde_prime2_theta" qtilde_prime2_theta = compute_qtilde_prime2_theta(choleskyΣθ, X, meanvv, Q, dQ, βhat_prime_theta, βhat_prime2_theta)
        #@timeit "Hθ_prime2" Hθ_prime2 = compute_Hθ_prime2(Bθ, Bθ_prime, Bθ_prime2, Σθ_inv_X, choleskyΣθ, X, Q, dQ)
        @timeit "m_prime2_theta" m_prime2_theta = compute_m_prime2_theta(Bθ, Bθ_prime, Bθ_prime2, choleskyΣθ, Σθ_prime, gλz, βhat, βhat_prime_theta, 
        βhat_prime2_theta, Hθ, Hθ_prime, Hθ_prime2, Q, dQ) 
        #@timeit "Dθ_prime2" Dθ_prime2 = compute_Dθ_prime2(choleskyΣθ, Bθ, Bθ_prime, Bθ_prime2, Eθ_prime2, Q, dQ)
        #@timeit "Cθ_prime2" Cθ_prime2 = compute_Cθ_prime2(Dθ_prime2, Hθ, Hθ_prime, Hθ_prime2, choleskyXΣX, dPinv, d2Pinv)
        #end
        @timeit "main deriv partial_theta" begin
        #compute derivative of main expression

        expr = z0 -> g(z0, λ) .- m
        qC = qtilde*Cθ 
        bilinearform = z0 -> 1 .+ expr(z0)'*(qC\(expr(z0)))
        detqC = det(qC) 
        qC_prime_theta =  qtilde_prime_theta .* Cθ + qtilde .* Cθ_prime
        AA = -0.5 * detqC^(-1/2) * tr(qC\(qC_prime_theta)) 
        qC_inv_prime_theta = Y -> - qC\(qC_prime_theta * (qC\Y))  
    
        dbilinearform = z0 -> -m_prime_theta'*(qC\(expr(z0))) .+ expr(z0)' * qC_inv_prime_theta(expr(z0)) .- expr(z0)'*(qC\m_prime_theta)
        EE = z0 -> bilinearform(z0)^(-(n-p+k)/2)
        FF = z0 -> detqC^(-1/2) * (-(n-p+k)/2) * (bilinearform(z0))^(-(n-p+k+2)/2)
        dT_pdf = z0 -> (cc*(AA*EE(z0) .+ FF(z0)*(dbilinearform(z0))))[1]
        dmain_pdf = z0 -> (cc*(AA*EE(z0) .+ FF(z0)*(dbilinearform(z0))))[1] * jac(z0)
        end
        @timeit "main second deriv partial_theta" begin
        #compute second derivative of main expression
        qC_prime2_theta = qtilde_prime2_theta .* Cθ + qtilde .* Cθ_prime2 + 2* Cθ_prime .* qtilde_prime_theta 
        qC_inv_prime2_theta = Y -> 2*(qC\(qC_prime_theta*(qC\(qC_prime_theta*(qC\Y))))) - qC\(qC_prime2_theta*(qC\Y))
        
        PP = 0.25 * (detqC)^(-1/2) * (tr(qC\qC_prime_theta))^2 
        QQ = -0.5*detqC^(-1/2) * tr(qC_inv_prime_theta(qC_prime_theta) + qC\(qC_prime2_theta)) 
        dAA = PP+QQ #second derivative of det(qC)^-1/2

        d2bilinearform = z0 -> (- m_prime2_theta'*(qC\expr(z0)) .- expr(z0)'*(qC\m_prime2_theta) .+ expr(z0)'*qC_inv_prime2_theta(expr(z0))
                                .+ 2*(m_prime_theta'*(qC\(m_prime_theta)) .- m_prime_theta'*(qC_inv_prime_theta(expr(z0))) .- expr(z0)'*(qC_inv_prime_theta(m_prime_theta)))
        )
        bformpower = z0 -> bilinearform(z0)^(-(n-p+k)/2)
        dbformpower = z0 -> -((n-p+k)/2)*bilinearform(z0)^(-(n-p+k+2)/2)*dbilinearform(z0)
        d2bformpower = z0 -> (n-p+k)/2 * (n-p+k+2)/2 * bilinearform(z0)^(-(n-p+k+4)/2)*dbilinearform(z0)^2 - (n-p+k)/2*bilinearform(z0)^(-(n-p+k+2)/2)*d2bilinearform(z0) 
        d2main_pdf = z0 -> (cc* (dAA*bformpower(z0) .+ 2*AA*dbformpower(z0) .+ detqC^(-1/2)*d2bformpower(z0)))[1] * jac(z0)
        end

        #derivatives of g(z0)-m/sqrt(qC/(n-p)) w.r.t theta for chain rule application
        locationscale = z0 -> expr(z0) ./ sqrt(qC/(n-p))
        dlocationscale = z0 -> - m_prime_theta ./ sqrt(qC ./ (n-p)) - 0.5 .* expr(z0) ./ (qC ./ (n-p))^1.5 .* qC_prime_theta/(n-p)
        d2locationscale = z0 ->  - m_prime2_theta ./ sqrt(qC/(n-p)) .+ m_prime_theta ./(qC/(n-p))^1.5 .* qC_prime_theta/(n-p) .+ expr(z0) * sqrt(n-p) .* (0.75*(qC)^(-5/2) .* (qC_prime_theta)^2 - 0.5 * qC^(-3/2) .* qC_prime2_theta)
        dmain_cdf = z0 -> vanillaT_pdf(expr(z0)/(sqrt(qC/(n-p)))) * dlocationscale(z0) #T_pdf(z0) .* dlocationscale(z0) #+ main_cdf(z0) .* (-0.5 ./ qC .* qC_prime_theta)
        
        #Below we write down some helper functions, which will allow us to compute derivatives using the built-in TDist CDF and PDF. 
        #Observe that d4' = d5 and d6' = d7
        d4 = z0 -> vanillaT_pdf(expr(z0)/(sqrt(qC/(n-p))))/sqrt(qC/(n-p)) 
        d5 = z0 -> dT_pdf(z0) 
        d6 = z0 -> vanillaT_pdf(expr(z0)/(sqrt(qC/(n-p))))
        d7 = z0 -> sqrt(qC)*(dT_pdf(z0)/sqrt(n-p) .+ 0.5*vanillaT_pdf(expr(z0)/(sqrt(qC/(n-p)))) ./ (qC)^(3/2) .* qC_prime_theta)
        
        d2main_cdf = z0 -> d7(z0) .* dlocationscale(z0) .+ vanillaT_pdf(expr(z0)/(sqrt(qC/(n-p))))*d2locationscale(z0)
        
        return [[main_pdf, dmain_pdf, d2main_pdf], [main_cdf, dmain_cdf, d2main_cdf]]
        #return ((main_pdf, dmain_pdf, d2main_pdf), (main_cdf, dmain_cdf, d2main_cdf, d4, d5, d6, d7)) 
        #return ((main_pdf, dmain_pdf, d2main_pdf), (main_cdf, dmain_cdf, d2main_cdf), (locationscale, dlocationscale, d2locationscale)) 
        
        #currently pdf and cdf are decoupled, so that calling one followe by the other results in 
        #repeated computation of certain quantities. However coupling them would result in extra  
        #computation if only one is needed.
    end
    #return (vec(Bθ_prime), vec(Bθ_prime2))
    return [[main_pdf], [main_cdf]]
    #return (dmain, d2main)
    #return (dbformpower, d2bformpower)
    #return (dbilinearform, d2bilinearform)
    #return (qC_prime_theta, qC_prime2_theta)
end
    #return (vec(qC_inv_prime_theta(I)), vec(qC_inv_prime2_theta(I)))
    #return (qtilde_prime_theta, qtilde_prime2_theta)
    #return (Cθ_prime, Cθ_prime2)
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
    #return (vec(qC_inv), vec(qC_inv_prime_theta))
    #return ([detqC^(-1/2)], [AA])
    #return (bilinearform, z0 -> BB(z0) + CC(z0) + DD(z0))

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
function partial_z0(θ::Float64, λ::Float64, train::trainingData{A, B}, test::testingData{A}, transforms, theta_params::Union{θ_params{A}, 
    Cholesky{Float64, A}, θ_param_derivs{A, Cholesky{Float64, A}}, Nothing} = nothing, type::String = "Gaussian") where A<:Array{Float64, 2} where B<:Array{Float64, 1}
    if type == "Gaussian" || type == "Turan"
        Eθ = theta_params.Eθ
        Σθ = theta_params.Σθ
        Bθ = theta_params.Bθ
        Dθ = theta_params.Dθ
        Hθ = theta_params.Hθ
        Cθ = theta_params.Cθ
        Σθ_inv_X = theta_params.Σθ_inv_X
        choleskyΣθ = theta_params.choleskyΣθ
        choleskyXΣX = theta_params.choleskyXΣX
    else
        throw(ArgumentError("Quadrature type not recognized."))
    end
    s = train.s; s0 = test.s0; X = train.X; X0 = test.X0; z = train.z; n = size(X, 1); p = size(X, 2); k = size(X0, 1)  #unpack 
    g = transforms.f #unpack
    dg = transforms.df
    dg2 = transforms.d2f
    gλz = g(z, λ)
    Σθinvgz = choleskyΣθ\gλz #precompute

    #intermediate quantities
    βhat = choleskyXΣX\(X'*Σθinvgz) 
    qtilde = (expr = gλz-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ * Σθinvgz + Hθ*βhat
    qC = qtilde*Cθ

    t = LocationScale(m[1], sqrt(qtilde[1]*Cθ[1]/(n-p)), TDist(n-p))
    #vanillat = LocationScale(0, 1, TDist(n-p))
    #vanillaT_pdf = z0 -> Distributions.pdf.(vanillat, z0)
    #T_pdf = z0 -> Distributions.pdf.(vanillat, g(z0, λ)) 
    
    function tdist(z0)
        gλz0 = g(z0, λ)
        if size(z0, 2)>1
            throw(ArgumentError("z0 must be 1-dimensional"))
        end
        Tz0 = Distributions.pdf.(t, gλz0) #.* abs.(dg(z0, λ))
        return Tz0 
    end

    function dtdist(z0)
        gλz0 = g(z0, λ)
        if size(z0, 2)>1
            throw(ArgumentError("z0 must be 1-dimensional"))
        end
        Tz0 = Distributions.pdf.(t, gλz0)
        return Tz0 .* (-(n-p+k)) .* ( gλz0 .- m) ./ (qC .+ (gλz0 .- m) .^2) .* dg(z0, λ)
    end

    function prob(z0)
        return tdist(z0) .* abs.(dg(z0, λ))
    end

    function dprob(z0)
        gλz0 = g(z0, λ)
        if size(z0, 2)>1
            throw(ArgumentError("z0 must be 1-dimensional"))
        end
        #Tz0 = Distributions.pdf.(vanillat, (gλz0 .- m) ./ sqrt(qC/(n-p)))
        Tz0 = Distributions.pdf.(t, gλz0)
        return (dg2(z0, λ) .* Tz0 .+ abs.(dg(z0, λ)) .* dtdist(z0))[1]
    end
    return [[dprob], [prob]]
    #return (tdist, dtdist)
    #############     Code for Multivariate z0    ##########
    #jac = z0 -> abs(reduce(*, map(x -> dg(x, λ), z0)))
    #djac = z0 ->(local n = length(z0);
    #            reshape(
    #            (local dd = map(x->dg(x, λ), z0); 
    #            local p = abs(reduce(*, dd));
    #            (f = i -> (dd[i] !=0 ? p/dd[i]*sign(dd[i])*(-1)^(((sign(dd[mod(i+1, n)+1])+1)/2)+1)* 
    #            dg2(z0[i], λ) : 0) ; 
    #            map(f, collect(1:n)))), n, 1))
    #qC = qtilde*Cθ
    #detqC = det(qC)
    #ccc = gamma((n-p+k)/2)/gamma((n-p)/2)/pi^(k/2)/detqC^(1/2) #constant
    #expr = z0 -> (g(z0, λ) .- m)'*(qC\(g(z0, λ) .- m))
    #dexpr = z0 -> (local n = length(z0);
    #                2 * Diagonal(dg(z0, λ)) * (qC\(g(z0, λ) .- m));
    #                )
    #expr3 = z0 -> ccc*jac(z0)*(1 .+ (g(z0, λ) .- m)'*(qC\(g(z0, λ) .- m)))^(-(n-p+k)/2)
    #dexpr3 = z0 -> ccc*(jac(z0)*(-(n-p+k)/2) * (1 .+ (g(z0, λ) .- m)'*(qC\(g(z0, λ) .- m)))^(-(n-p+k+2)/2) * 2 * Diagonal(dg(z0, λ)) * (qC\(g(z0, λ) .- m)) 
    #                + djac(z0)*(1 .+ (g(z0, λ) .- m)'*(qC\(g(z0, λ) .- m)))^(-(n-p+k)/2))
    ######################################################
end


"""
Compute deriative of p(z0|theta, lambda, z) with respect to s
"""
function partial_s(θ::Float64, λ::Float64, train::trainingData{A, B}, test::testingData{A}, transforms, theta_params::Union{θ_params{A}, 
    Cholesky{Float64, A}, θ_param_derivs{A, Cholesky{Float64, A}}, Nothing} = nothing, type::String = "Gaussian") where A<:Array{Float64, 2} where B<:Array{Float64, 1}
    if type == "Gaussian" || type == "Turan"
        Eθ = theta_params.Eθ
        Σθ = theta_params.Σθ
        Bθ = theta_params.Bθ
        Dθ = theta_params.Dθ
        Hθ = theta_params.Hθ
        Cθ = theta_params.Cθ
        Σθ_inv_X = theta_params.Σθ_inv_X
        choleskyΣθ = theta_params.choleskyΣθ
        choleskyXΣX = theta_params.choleskyXΣX
    else
        throw(ArgumentError("Quadrature type not recognized."))
    end
    s = train.s; s0 = test.s0; X = train.X; X0 = test.X0; z = train.z; n = size(X, 1); p = size(X, 2); k = size(X0, 1)  #unpack 
    g = transforms.f #unpack
    dg = transforms.df
    dg2 = transforms.d2f
    gλz = g(z, λ)
    Σθinvgz = choleskyΣθ\gλz #precompute

    #intermediate quantities
    βhat = choleskyXΣX\(X'*Σθinvgz) 
    qtilde = (expr = gλz-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ * Σθinvgz + Hθ*βhat
    qC = qtilde*Cθ

    t = LocationScale(m[1], sqrt(qtilde[1]*Cθ[1]/(n-p)), TDist(n-p))
    #vanillat = LocationScale(0, 1, TDist(n-p))
    #vanillaT_pdf = z0 -> Distributions.pdf.(vanillat, z0)
    #T_pdf = z0 -> Distributions.pdf.(vanillat, g(z0, λ)) 

    d = maximum(size(s0, 1), size(s0, 2))
    sdiff = repeat(s0, n, 1) .- s
    gradB = zeros(n, d)
    gradB = -θ * Bθ .* (sdiff)

    gradE = [1]

    gradD = -2 * θ * gradB' * (choleskyΣθ\Bθ) #in the future, store sigma inv B

    HessianD1 = -2 * θ^2 *  gradB' * (choleskyΣθ\gradB) 
    HessianD2 = zeros(d, d)
    for i = 1:d
        for j = 1:i
            HessianD2[i, j] = -θ^2 * choleskyΣθ\ (gradB[:, i] .* sdiff[:, i] .+ Bθ) 
        end
    end

    



end


"""
Compute p(theta, lambda| z), possibly in addition to derivatives with respect to theta. Differs from paper in that Jacobian term computation is delayed 
until the matrix of weights is formed. 
"""
#function posterior_theta(θ::Float64, λ::Float64, pθ, dpθ, dpθ2, pλ, setting::setting{Array{Float64,2}, Array{Float64, 1}}, theta_params::Union{θ_params{Array{Float64, 2}, 
function posterior_theta(θ::Float64, λ::Float64, priorθ, priorλ, train::trainingData{A, B}, test::testingData{A}, transforms, theta_params::Union{θ_params{A, 
    Cholesky{Float64, A}}, θ_param_derivs{A, Cholesky{Float64,A}}, Nothing} = nothing, type::String="Gaussian") where A<:Array{Float64, 2} where B<:Array{Float64, 1}
    if theta_params === nothing
        #println("WARNING: recompute theta_params in partial_theta")
        theta_params = funcθ(θ, train, test, type)
        #(Eθ, Σθ, Bθ, Dθ, Hθ, Cθ, Eθ_prime, Eθ_prime2, Σθ_prime, Σθ_prime2, Bθ_prime, Bθ_prime2, choleskyΣθ, choleskyXΣX) = getθ_Params(theta_params)
        #(s, s0, X, X0, z,  n, p, k) = getSettingParams(setting)
    end
    if type == "Gaussian" || type == "Turan"
        Eθ = theta_params.Eθ
        Σθ = theta_params.Σθ
        Bθ = theta_params.Bθ
        Dθ = theta_params.Dθ
        Hθ = theta_params.Hθ
        Cθ = theta_params.Cθ
        Σθ_inv_X = theta_params.Σθ_inv_X
        choleskyΣθ = theta_params.choleskyΣθ
        choleskyXΣX = theta_params.choleskyXΣX
    else
        throw(ArgumentError("Quadrature type undefined. Please enter \"Gaussian\" or \"Turan\" for last arg."))
    end
    if type == "Turan" #load higher derivatives
        Eθ_prime = theta_params.Eθ_prime
        Eθ_prime2 = theta_params.Eθ_prime2
        Σθ_prime = theta_params.Σθ_prime
        Σθ_prime2 = theta_params.Σθ_prime2
        Bθ_prime = theta_params.Bθ_prime
        Bθ_prime2 = theta_params.Bθ_prime2
        Dθ_prime = theta_params.Dθ_prime
        Dθ_prime2 = theta_params.Dθ_prime2
        Hθ_prime = theta_params.Hθ_prime
        Hθ_prime2 = theta_params.Hθ_prime2
        Cθ_prime = theta_params.Cθ_prime
        Cθ_prime2 = theta_params.Cθ_prime2
        tripleΣ = theta_params.tripleΣ
    end
    #unpack args 
    pθ = priorθ.f; dpθ = priorθ.df; dpθ2 = priorθ.d2f; pλ = priorλ.f
    s = train.s; s0 = test.s0; X = train.X; X0 = test.X0; z = train.z; n = size(X, 1); p = size(X, 2); k = size(X0, 1) 
    g = transforms.f; dg = transforms.df
    #compute auxiliary quantities 
    gλz = g(z, λ)
    βhat = choleskyXΣX\(X'*(choleskyΣθ\gλz)) 
    qtilde = (expr = gλz-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ*(choleskyΣθ\gλz) + Hθ*βhat 
    # Σθ_inv_X = Σθ\X # XZ: already computed in theta_params
    meanvv = gλz - X*βhat 

    EXPR1 = det(choleskyΣθ)^(-1/2)
    #println("det: ", EXPR1)
    #println("det Sigma theta ^-1/2: ", EXPR1)
    EXPR2 = det(choleskyXΣX)^(-1/2)
    #println("det XsigmaX: ", EXPR2)
    EXPR3 = qtilde^(-(n-p)/2) 
    #println("qtilde: ", qtilde)
    EXPR4 = pθ(θ)*pλ(λ)
    #println("EXPR4: ", EXPR4)
    #main = EXPR1*EXPR2*EXPR3*EXPR4
    #main = EXPR2*EXPR3*EXPR4
    main = EXPR2 * EXPR4
    if type=="Turan"
        #compute betahat_prime_theta
        expr_mid = X'*(choleskyΣθ\Σθ_prime)*(Σθ_inv_X)
        βhat_prime_theta = compute_betahat_prime_theta(choleskyΣθ, choleskyXΣX, expr_mid, Σθ_prime, X, gλz, Σθ_inv_X)

        #compute qtilde_prime_theta
        qtilde_prime_theta  = compute_qtilde_prime_theta(gλz, X, βhat, βhat_prime_theta, choleskyΣθ, Σθ_prime)
    
        trΣθqΣθ_prime = tr(choleskyΣθ\Σθ_prime) #precompute 
        dEXPR1 = -0.5 * det(choleskyΣθ)^(-1/2)*trΣθqΣθ_prime
        XΣdΣΣX = Σθ_inv_X' * Σθ_prime * Σθ_inv_X  #precompute
        trexpr = tr(choleskyXΣX\(-XΣdΣΣX))#precompute
        dEXPR2 = -0.5*det(choleskyXΣX)^(-1/2)*trexpr
        dEXPR3 = -((n-p)/2)*qtilde^(-(n-p+2)/2)*qtilde_prime_theta
        dEXPR4 = dpθ(θ)*pλ(λ)

        dmain = (dEXPR1 * EXPR2 * EXPR3 * EXPR4 .+ dEXPR2*EXPR1*EXPR3*EXPR4 .+ dEXPR3*EXPR1*EXPR2*EXPR4 .+ dEXPR4*EXPR1*EXPR2*EXPR3)[1]

        #-====================== dmain2 (second derivative)===============================
        d2EXPR1 = 0.25 * EXPR1 * trΣθqΣθ_prime^2 -0.5 * EXPR1* tr(choleskyΣθ\Σθ_prime2 - choleskyΣθ\(Σθ_prime*(choleskyΣθ\Σθ_prime)))

        dQ = Y -> (choleskyΣθ\((Σθ_prime2*(choleskyΣθ\Y)))) - 2*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Y))))))
        dPinv = Y -> choleskyXΣX\(X'*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(X*(choleskyXΣX\Y))))))

        d2EXPR2 = 0.25 * EXPR2 * trexpr^2 .+ 0.5*EXPR2*tr(dPinv(XΣdΣΣX) .+ choleskyXΣX\(X'*dQ(X)))

        Q = Y -> (choleskyΣθ\(Σθ_prime*(Σθ\(Y))))
        βhat_prime2_theta = compute_βhat_prime2_theta(choleskyXΣX, choleskyΣθ, expr_mid, X, Q, dQ, gλz)
        qtilde_prime2_theta = compute_qtilde_prime2_theta(choleskyΣθ, X, meanvv, Q, dQ, βhat_prime_theta, βhat_prime2_theta)
        d2EXPR3 = ((n-p)/2)*((n-p+2)/2)*qtilde^(-(n-p+4)/2)*qtilde_prime_theta^2 - (n-p)/2 * qtilde^(-(n-p+2)/2)*qtilde_prime2_theta
        d2EXPR4 = dpθ2(θ)*pλ(λ)
        d2main = (d2EXPR1*EXPR2*EXPR3*EXPR4 .+ d2EXPR2*EXPR1*EXPR3*EXPR4 .+ d2EXPR3*EXPR1*EXPR2*EXPR4 .+ d2EXPR4*EXPR1*EXPR2*EXPR3 
                    .+ 2*(dEXPR1*dEXPR2*EXPR3*EXPR4 .+ dEXPR1*EXPR2*dEXPR3*EXPR4 .+ dEXPR1*EXPR2*EXPR3*dEXPR4 .+ EXPR1*dEXPR2*dEXPR3*EXPR4
                    .+ EXPR1*dEXPR2*EXPR3*dEXPR4 .+ EXPR1*EXPR2*dEXPR3*dEXPR4))[1]
        if false #debug
            println("===============================================================")
            println("theta");println(θ);println("betahat");println(βhat);println("glambdaz");println(gλz);println("qtilde")
            println(qtilde);println("m");println(m);println("expr_mid");println(expr_mid);println("βhat_prime2_theta")
            println(βhat_prime2_theta);println(" qtilde_prime2_theta");println(qtilde_prime2_theta)
            println("pθ(θ)"); println(pθ(θ)); println("pλ(λ)");println(pλ(λ));println("EXPR1");println(EXPR1);println("EXPR2");println(EXPR2)
            println("EXPR3"); println(EXPR3);println("EXPR4");println(EXPR4);println("dEXPR1"); println(dEXPR1)
            println("dEXPR2"); println(dEXPR2);println("dEXPR3");println(dEXPR3);println("dEXPR4")
            println(dEXPR4); println("d2EXPR1");println(d2EXPR1);println("d2EXPR2");println(d2EXPR2)
            println("d2EXPR3");println(d2EXPR3);println("d2EXPR4");println(d2EXPR4);println("main")
            println(main);println("dmain");println(dmain);println("d2main");println(d2main)
            println("===============================================================")
        end
        return (main, dmain, d2main)
    elseif type == "Gaussian"   
        return main 
    else 
        throw(ArgumentError("Quadrature type undefined. Please use \"Gaussian\" or \"Turan\" for last arg"))
    end
 
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
function checkDerivative(f, df, x0, hessian = nothing, first = 3, last = 12, num = 10)
    f0 = f(x0)
    df0 = df(x0) 
    if hessian!=nothing 
        d2f0 = hessian(x0)
    end

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
       
            if hessian!=nothing
                inc = h[i] * dx
                A[i] = norm((fi .- f0) .- df0' * inc .- 0.5* inc' * d2f0 * inc)
            else
                try
                A[i] = norm((fi .- f0) .- df0' * (h[i] * dx))
            catch DimensionMismatch #we catch the case when f: R^1 -> R^n, in which case  df0'*dx will yield an error
                #println("caught in check deriv")
                A[i] = norm((fi .- f0) .- df0 .* (h[i] * dx))
            end
           
        end
    end
    #println(A)
    #println(h)

    return (log.(h), log.(A))
end

