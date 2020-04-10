#include("../model0.jl")
#include("../computation/buffers0.jl")

using Cubature

"""
Compute cdf, pdf, and pdf_deriv of T-distribution
"""
function comp_tdist(btg::btg, θ::Array{T, 1}, λ::Array{T, 1}) where T<:Float64
    trainingData = btg.trainingData
    g = btg.g #nonlinear transform, e.g. BoxCox
    invg = (x, λ) -> inverse(btg.g, x, λ)
    (_, Σθ_inv_X, choleskyΣθ, _) = unpack( btg.train_buffer_dict[θ])
    (x, Fx, y, _, n, p) = unpack(trainingData) #unpack training data

    gλy = g(y, λ) #apply nonlinar transform to observed labels y
    βhat = (Fx'*Σθ_inv_X)\(Fx'*(choleskyΣθ\gλy)) 
    qtilde = (expr = gλy-Fx*βhat; expr'*(choleskyΣθ\expr)) 
    meanvv = gλy - Fx*βhat
    Σθ_inv_y = (choleskyΣθ\gλy)

    dg = (y, λ) -> partialx(g, y, λ) #derivative w.r.t z
    dg2 = (y, λ) -> partialxx(g, y, λ) #second derivative w.r.t z
    jac = x -> abs(reduce(*, map(z -> dg(z, λ), x))) #Jacobian function
    
    function compute_qmC(x0, Fx0)#assume q and C are 1x1
        x0 = reshape(x0, 1, length(x0)) #reshape into 1 x d vector
        update!(btg.testingData, x0, Fx0)#update testing data with x0, Fx0
            
        x1, y1 = getDimension(btg.trainingData), getDimension(btg.testingData)
        x2, y2 = getCovDimension(btg.trainingData), getCovDimension(btg.testingData)
        #if (~ (x1==y1) || ~ (x2 == y2))
            # println("dimension of train is $x1 but dimension of test is $y1")
            # println("covariance dimensions of train is $x2 but covariance dimension of test is $y2")
        #end
        update!(btg.train_buffer_dict[θ], btg.test_buffer_dict[θ], btg.trainingData, btg.testingData)#update θ-testing buffer with recomputed Bθ, Hθ, Cθ,...
        (_, Bθ, _, _, Hθ, Cθ) = unpack(btg.test_buffer_dict[θ])
        #println("Bθ: ", Bθ)
        #println("Hθ: ", Hθ)
        #println("Cθ: ", Cθ)
        m = Bθ*(choleskyΣθ\gλy) + Hθ*βhat #recompute mean
        #println("m: ", m[1])
        qC = qtilde[1]*Cθ[1] #both q and C are 1x1 for single-point prediction
        sigma_m = qC/(n-p-2) + m[1]^2 # E[T_i^2] for quantile estimation
        return m[1], qtilde[1], Cθ[1], βhat  #sigma_m
    end

    function compute(f, x0, Fx0, y0)#updates testingData and test_buffer, but leaves train_buffer and trainingData alone
        m, q, C = compute_qmC(x0, Fx0)
        qC = q*C
        t = LocationScale(m, sqrt(qC/(n-p)), TDist(n-p)) #avail ourselves of built-in tdist
        return f(y0, t, m, qC) # return location parameter to utilize T mixture structure
    end
    
    main_pdf = (y0, t, m, qC) -> (Distributions.pdf.(t, g(y0, λ)) * jac(y0))
    main_cdf = (y0, t, m, qC) -> (Distributions.cdf.(t, g(y0, λ))) 
    main_pdf_deriv_helper = (y0, t, m, qC) -> (k = size(qC, 1); gλy0 = g(y0, λ); Ty0 = Distributions.pdf.(t, gλy0);
                            Ty0 .* (-(n-p+k)) .* ( gλy0 .- m) ./ (qC .+ (gλy0 .- m) .^2) .* dg(y0, λ)) #this is a stable computation of the derivative of the tdist
    main_pdf_deriv = (y0, t, m, qC) -> (gλy0 = g(y0, λ);  
                    Ty0 = Distributions.pdf.(t, gλy0); (dg2(y0, λ) .* Ty0 .+ abs.(dg(y0, λ)) .* main_pdf_deriv_helper(y0, t, m, qC))[1])
    pdf_deriv = (x0, Fx0, y0) -> compute(main_pdf_deriv, x0, Fx0, y0)
    pdf = (x0, Fx0, y0) -> compute(main_pdf, x0, Fx0, y0)
    cdf = (x0, Fx0, y0) -> compute(main_cdf, x0, Fx0, y0)

    # parallel to compute, except used to compute derivatives w.r.t location
    # also calls compute_qmC and hence updates test buffers
    # Comes after pdf, cdf, pdf_deriv computations, because those are needed for current computation 
    function compute_location_derivs(x0, Fx0, y0)
        m, q, C, βhat = compute_qmC(x0, Fx0)
        qC = q*C
        k = size(qC, 1)
        gλy0 = btg.g(y0, λ)
        arg = ((gλy0 .- m)/sqrt(qC/(n-p)))[1] #1 x 1
        vanillat = LocationScale(0, 1, TDist(n-p))
        cdf_eval =  Distributions.cdf.(vanillat, arg)
        cdf_deriv = Distributions.pdf.(vanillat, arg) #chain rule term is computed later
        cdf_second_deriv = -(n-p+k) * arg/(1 + arg^2) * Distributions.pdf.(vanillat, arg) #chain rule term computed later
        # t = LocationScale(m, sqrt(qC/(n-p)), TDist(n-p)) #avail ourselves of built-in tdist
        # #(C, jacC) = compute_higher_derivs(btg, θ, x0, Fx0, y0)
        # pdf_eval = Distributions.pdf.(vanillat, arg)
        # cdf_eval = Distributions.cdf.(vanillat, arg)
        # pdf_deriv_eval = main_pdf_deriv_location_helper(y0, t, m, qC)
        (func, jacobian, hessian) = compute_BO_derivs(btg, θ, λ, x0, Fx0, y0, m, q, βhat, Σθ_inv_y, cdf_deriv, cdf_second_deriv, cdf_eval)
    end

    cdf_prime_loc = (x0, Fx0, y0) -> compute_location_derivs(x0, Fx0, y0)
    
    #derivatives w.r.t s + Hessian for augmented vector (y0, x0) for Bayesian optimization

    function main_cdf_prime_s() 
        #used to compute gradient and Hessian using necessary ingredients
    end
    #cdf_prime_s = (x0, Fx0, y0) -> compute_higher_derivs(main_cdf_prime_s, ...)#gradient of B(s), X0(s), D(s), H(s)
    #cdf_hessian = (x0, Fx0, y0) -> compute_higher #use results from cdf_prime_s and main_pdf_deriv

    m = (x0, Fx0) -> hquadrature(y0 -> y0 * pdf(x0, Fx0, y0), 0, 2)[1]
    Ex2 = (x0, Fx0) -> hquadrature(y0 -> y0^2 * pdf(x0, Fx0, y0), 0, 2)[1]
    # m = (x0, Fx0) -> compute_qmC(x0, Fx0)[1] 
    # sigma_m = (x0, Fx0) -> compute_qmC(x0, Fx0)[3] 

    # compute quantile q for each component
    function q_fun(x0, Fx0, quant)
        m, q, C = compute_qmC(x0, Fx0)
        t = LocationScale(m, sqrt(q*C/(n-p)), TDist(n-p))
        return Distributions.quantile(t, quant)
    end
    # q_fun = (x0, Fx0, q) -> (m, q, C = compute_qmC(x0, Fx0); invg(sqrt(q*C/(n-p))*tdistinvcdf(n-p, q)+m, λ))
    
    return (pdf_deriv, pdf, cdf, cdf_prime_loc, m, Ex2, q_fun)
end


"""
* Assumes Fx0 is linear polynomial mean basis 
* Assume btg testing_buffer has been updated using x0, Fx0, y0, i.e. compute_qmc has been called
"""
function compute_BO_derivs(btg::btg, θ, λ, x0, Fx0, y0, m, q, βhat, Σθ_inv_y, cdf_deriv, cdf_second_deriv, cdf_eval)
    x0 = reshape(x0, 1, length(x0))
    #unpack pertinent quantities
    (_, Σθ_inv_X, choleskyΣθ, choleskyXΣX) = unpack(btg.train_buffer_dict[θ])
    (x, Fx, y, d, n, p) = unpack(btg.trainingData) 
    (Eθ, Bθ, ΣθinvBθ, Dθ, Hθ, Cθ) = unpack(btg.test_buffer_dict[θ])
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(compute gradients)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #check if theta is an array of length d
    #println("size of Fx: ", size( Fx))
    #println("size of H: ", size( Hθ))
    #println("size of B: ", size(Bθ))
    @assert typeof(θ)<:Array{T, 1} where T 
    @assert length(θ) ==d
    S = repeat(x0, n, 1) .- x # n x d 
    #println("size of S: ", size(S))
    jacB =   diagm(Bθ[:]) * S * diagm(- θ[:]) #n x d
    #println("size of jacB: ", size(jacB))
    jacD = -2* jacB' * (choleskyΣθ \ Bθ') #d x 1
    #println("size of jacD: ", size(jacD))
    #assuming linear polynomial basis
    jacFx0 = vcat(zeros(1, d), diagm(ones(d))) #p x d
    #println("size of Fx0: ", size(Fx0))
    jacH = jacFx0' - jacB' * Σθ_inv_X #d x p
    #println("size of jacH: ", size(jacH))
    jacC = jacD + 2 * jacH * (choleskyXΣX \ Hθ') #d x 1
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(compute hessian of D)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ΣinvjacB  = choleskyΣθ\jacB 
    W = zeros(d, d)
    Stheta = S * diagm(- θ[:]) # n x d
    for i = 1:d
        for j = 1:d
            arg = Bθ' .* Stheta[:, i:i] .* Stheta[:, j:j] 
            if i==j   
                eq = Bθ * (choleskyΣθ \ (-θ[i] .* Bθ'))
            else
                eq = 0
            end
            W[i, j] = (Bθ * (choleskyΣθ\arg) .+ eq)[1]
        end
    end
    hessD = -2*jacB' * ΣinvjacB - 2*W #d x d

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(compute hessian of C)~~~~~~~~~~~~~~~~~~~~~~~~~~~
    V = zeros(d, d)
    for i = 1:d
        for j = 1:d
            arg = Bθ' .* Stheta[:, i:i] .* Stheta[:, j:j] 
            if i==j   
                eq = (-θ[i] .* Bθ')
            else
                eq = 0
            end
            #println("size arg: ", size(arg))
            #println("size eq: ", size(eq))
            # println(arg .+ eq)
            # println((Fx' * (choleskyΣθ \ (arg .+ eq))))
            # println((choleskyXΣX \ (Fx' * (choleskyΣθ \ (arg .+ eq)))))
            # println(Hθ)
            V[i, j] =  (Hθ * (choleskyXΣX \ ( - Fx' * (choleskyΣθ \ (arg .+ eq)))))[1]
        end
    end
    hessC = hessD + 2*jacH*(choleskyXΣX\jacH') + 2*V #d x d

    #~~~~~~~~~~~~~~~~~~~~~~~~~(compute derivative and hessian of m)~~~~~~~~~~~~~~~~~~~~~~~~~~
    jacm = jacB' * Σθ_inv_y + jacH * βhat  # d x 1

    hess_m = zeros(d, d)
    for i = 1:d
        for j = 1:d
            arg = Bθ' .* Stheta[:, i:i] .* Stheta[:, j:j] 
            if i==j   
                eq = (-θ[i] .* Bθ')
            else
                eq = 0
            end
            #println()
            #println("sigma inv y: ", size(Σθ_inv_y))
            #println("size arg .+ eq: ", size(arg .+ eq))
            expr1 = dot(Σθ_inv_y,  (arg .+ eq))
            expr2 = dot(( - Fx' * (choleskyΣθ \ (arg .+ eq))), βhat)
            #println("size expr1: ", size(expr1))
            #println("size expr2: ", size(expr2))
            hess_m[i, j] = (expr1 + expr2)[1] #d_i d_j H
        end
    end

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(compute gradient of CDF G(u, s) w.r.t s)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #G(u, s)
    gλy0 = btg.g(y0, λ[1])
    #we need to define covariance (sigma here ...)
    #sigma is covariance sqrt(qC/(n-p))
    qC = (q .* Cθ[1])
    sigma = sqrt(qC/(n-p)) #covariance
    dsigma = sqrt(q/(n-p)) .* 0.5 * Cθ[1]^(-1/2) .* jacC 
    hess_sigma = zeros(d, d)
    for i = 1:d
        for j = 1:d
            hess_sigma[i, j] = -0.25 * Cθ[1]^(-3/2) .* jacC[i]jacC[j] + 0.5 * Cθ[1]^(-1/2) * hessC[i, j]
        end
    end

    function Y(i, sigma, dsigma) 
        return - jacm[i]/sigma + (gλy0 - m)/sigma^2 * dsigma[i]    
    end
    function R(i, j, sigma, dsigma, hess_sigma) #assume jacm  and hess_m defined already
        e1 = -hess_m[i, j]/sigma 
        e2 = jacm[j] * dsigma[i] / sigma^2
        e3 = - jacm[i] * dsigma[j] / sigma^2
        e4 = - 2*(gλy0 - m) * dsigma[j]/sigma^3
        e5 = (gλy0 - m)*hess_sigma[i, j] / sigma^2
        return e1 + e2 + e3 + e4 + e5
    end
    G = cdf_eval 
    jacG = zeros(1, d) #1 x d
    for i = 1:d 
        jacG[i] = (cdf_deriv .* Y(i, sigma, dsigma))[1]
    end

    hess_G = zeros(d, d)
    for i = 1:d
        for j = 1:d
            expr1 = cdf_second_deriv * Y(i, sigma, dsigma) * Y(j, sigma, dsigma)
            expr2 = cdf_deriv * R(i, j, sigma, dsigma, hess_sigma)
            hess_G[i, j] = (expr1 .+ expr2)[1]
        end
    end
    #println("size G: ", size(G))
    #println("size jacG: ", size(jacG))
    #println("size hess_G: ", size(hess_G))

    # println("cdf_eval: ", cdf_eval)
    # println("cdf_deriv: ", cdf_deriv)
    # println("cdf_second_deriv: ", cdf_second_deriv)
    # return (cdf_eval, cdf_deriv, cdf_second_deriv)


    return (G, jacG, hess_G)
    #return (m, jacm', hess_m)
    #return (Bθ*(choleskyΣθ\Bθ'), 2*Bθ*(choleskyΣθ\jacB), 2 * jacB' * (choleskyΣθ\ jacB))
    #return (Dθ, jacD', hessD)
    #return (Cθ, jacC', hessC) #dimensions of jacobian needs to be correct with respect to input and output dimensinos
    
end