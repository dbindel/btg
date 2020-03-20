"""
Computes T-Distribution PDF and CDF stably using built-in function.
Returns:
    p(z0 | theta, lambda, z) 
    P(z0 | theta, lambda, z) 
"""
function likelihood(θ, λ, train, test, theta_params::Union{θ_params{Array{Float64, 2}, 
    Cholesky{Float64,Array{Float64, 2}}}, Nothing}=nothing, type = "Gaussian")
    s = train.s; s0 = test.s0; X = train.X; X0 = test.X0; z = train.z; n = size(X, 1); p = size(X, 2); k = size(X0, 1)  #unpack 
    if theta_params==nothing; theta_params = funcθ(θ, train, test, type); end
    g = boxCox #boxCox by default
    Bθ = theta_params.Bθ
    Cθ = theta_params.Cθ
    Hθ = theta_params.Hθ
    Σθ_inv_X = theta_params.Σθ_inv_X
    choleskyΣθ = theta_params.choleskyΣθ
    βhat = (X'*(choleskyΣθ\X))\(X'*(choleskyΣθ\g(z, λ)))
    qtilde = (expr = g(z, λ)-X*βhat; expr'*(choleskyΣθ\expr)) 
    m = Bθ*(choleskyΣθ\g(z, λ)) + Hθ*βhat  
    t = LocationScale(m[1], sqrt(qtilde[1]*Cθ[1]/(n-p)), TDist(n-p))
    return (z0 -> Distributions.pdf(t, g(z0, λ)), z0 -> Distributions.cdf(t, g(z0, λ)))
end