include("../model0.jl")

"""
Compute cdf, pdf, and pdf_deriv of T-distribution
"""
function comp_tdist(btg::btg, θ::Array{T, 1}, λ::Array{T, 1}) where T<:Float64
    trainingData = btg.trainingData
    train_buffer = btg.train_buffer_dict[θ]
    g = btg.g #nonlinear transform, e.g. BoxCox

    (_, Σθ_inv_X, choleskyΣθ, _) = unpack(train_buffer)
    (x, Fx, y, _, n, p) = unpack(trainingData) #unpack training data

    gλy = g(y, λ) #apply nonlinar transform to observed labels y
    βhat = (Fx'*Σθ_inv_X)\(Fx'*(choleskyΣθ\gλy)) 
    qtilde = (expr = gλy-Fx*βhat; expr'*(choleskyΣθ\expr)) 
    meanvv = gλy - Fx*βhat

    dg = (y, λ) -> partialx(g, y, λ) #derivative w.r.t z
    dg2 = (y, λ) -> partialxx(g, y, λ) #second derivative w.r.t z
    jac = x -> abs(reduce(*, map(z -> dg(z, λ), x))) #Jacobian function
    
    function compute(f, x0, Fx0, y0)#updates testingData and test_buffer, but leaves train_buffer and trainingData alone
        update!(btg.testingData, x0, Fx0)#update testing data with x0, Fx0
        update!(train_buffer, btg.test_buffer_dict[θ], btg.trainingData, btg.testingData)#update θ-testing buffer with recomputed Bθ, Hθ, Cθ,...
        (_, Bθ, _, _, Hθ, Cθ) = unpack(btg.test_buffer_dict[θ])
        m = Bθ*(choleskyΣθ\gλy) + Hθ*βhat #recompute mean
        qC = qtilde[1]*Cθ[1] #both q and C are 1x1 for single-point prediction
        sigma_m = qC/(n-p-2) + m^2 
        t = LocationScale(m[1], sigma, TDist(n-p)) #avail ourselves of built-in tdist
        return f(y0, t, m, qC), m, sigma_m # return location parameter to utilize T mixture structure
    end
    
    main_pdf = (y0, t, m, qC) -> (Distributions.pdf.(t, g(y0, λ)) * jac(y0))

    main_cdf = (y0, t, m, qC) -> (Distributions.cdf.(t, g(y0, λ))) 

    main_pdf_deriv_helper = (y0, t, m, qC) -> (k = size(qC, 1); gλy0 = g(y0, λ); Ty0 = Distributions.pdf.(t, gλy0);
                            Ty0 .* (-(n-p+k)) .* ( gλy0 .- m) ./ (qC .+ (gλy0 .- m) .^2) .* dg(y0, λ)) #this is in fact a stable computation
    main_pdf_deriv = (y0, t, m, qC) -> (gλy0 = g(y0, λ); 
                    Ty0 = Distributions.pdf.(t, gλy0); (dg2(y0, λ) .* Ty0 .+ abs.(dg(y0, λ)) .* main_pdf_deriv_helper(y0, t, m, qC))[1])
    pdf_deriv = (x0, Fx0, y0) -> compute(main_pdf_deriv, x0, Fx0, y0)[1]
    pdf = (x0, Fx0, y0) -> compute(main_pdf, x0, Fx0, y0)[1]
    m = (x0, Fx0) -> compute(main_pdf, x0, Fx0, y0)[2]
    sigma_m = (x0, Fx0) -> compute(main_pdf, x0, Fx0, y0)[3]
    cdf = (x0, Fx0, y0) -> compute(main_cdf, x0, Fx0, y0)[1]

    return (pdf_deriv, pdf, cdf, m, sigma_m)
end
