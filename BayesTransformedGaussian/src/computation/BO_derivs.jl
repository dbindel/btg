"""
* Assumes Fx0 is linear polynomial mean basis 
* Assume btg testing_buffer has been updated using x0, Fx0, y0, i.e. compute_qmc has been called

NOTE:θ theta needs to be passed as an array
"""
function compute_BO_derivs(train_buffer, train_data, test_buffer, θ, λ, x0, Fx0, y0, m, q, βhat, Σθ_inv_y, cdf_deriv, cdf_second_deriv, cdf_eval)
    x0 = reshape(x0, 1, length(x0))
    #unpack pertinent quantities
    Σθ_inv_X  = btg.train_buffer_dict[θ].Σθ_inv_X
    choleskyΣθ = btg.train_buffer_dict[θ].choleskyΣθ
    choleskyXΣX = btg.train_buffer_dict[θ].choleskyXΣX
    (x, Fx, y, d, n, p) = unpack(btg.trainingData) 
    (Eθ, Bθ, ΣθinvBθ, Dθ, Hθ, Cθ) = unpack(btg.test_buffer_dict[θ])
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(compute gradients)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #check if theta is an array of length d
    #println("n (number data points in trainingData): ", n)
    #println("d (number length scales): ", d)
    #println("size of Fx: ", size( Fx))
    #println("size of H: ", size( Hθ))
    #println("size of B: ", size    (Bθ))
    function intermediates(x0)::Tuple{Array{T}, Array{T}, Array{T}, Array{T}} where T<:Real
        @assert typeof(θ)<:Array{T, 1} where T 
        @assert length(θ) ==d
        S = repeat(x0, n, 1) .- x # n x d 
        #println("size of S: ", size(S))
        jacB =   diagm(Bθ[:]) * S * diagm(- θ[:]) #n x d
        #println("size of jacB: ", size(jacB))
        #println("size of Bθ: ", size(Bθ))
        #println("size of choleskyΣθ: ", size(choleskyΣθ))
        #println("choleskyΣθ inv Bθ': ", size(choleskyΣθ \ Bθ'))
        jacD = -2* jacB' * (choleskyΣθ \ Bθ') #d x 1
        #println("size of jacD: ", size(jacD))
        #assuming linear polynomial basis
        jacFx0 = vcat(zeros(1, d), diagm(ones(d))) #p x d
        #println("size of Fx0: ", size(Fx0))
        jacH = jacFx0' - jacB' * Σθ_inv_X #d x p
        #println("size of jacH: ", size(jacH))
        jacC = jacD + 2 * jacH * (choleskyXΣX \ Hθ') #d x 1
        jacm = jacB' * Σθ_inv_y + jacH * βhat  # d x 1
        return (jacB, jacD, jacFx0, jacC)
    end

    function hessians()
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
        #~~~~~~~~~~~~~~~~~~~~~~~~~(compute hessian of m)~~~~~~~~~~~~~~~~~~~~~~~~~~
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