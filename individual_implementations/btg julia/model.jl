using LinearAlgebra
using Distributions
using SpecialFunctions
using PDMats
using Printf

include("kernel.jl")
include("examples.jl")

"""
Define prediction/inference problem by supplying known parameters, including design matrices, 
observation locations, prediction locations, nonlinear transform, prior marginals on theta and lambda,
and anticipated ranges for theta and lambda. Returns Bayesian predictive density function, 
namely the integral of p(z_0| theta, lambda, z)*p(theta, lambda | z) over Theta x Lambda
(See Equations 8 and 12)

INPUTS: 
X: k x p design matrix 
X0: n x p design matrix
s: observed locations
s0: prediction locations
g: nonlinear transform parametrized by lambda. Usage: g(lambda, z)
gprime: derivative of g. Usage: gprime(z, lambda)
ptheta: prior marginal on theta; function handle 
plambda: prior marginal on lambda; function handle
theta: structural parameter
lambda: transform parameter
z: observed random vector
z0: unobserved random vector

OUTPUTS:
probability density function z0 -> f(z_0|z) 
cumulative density function z0 -> F(z0|z)
"""

function model(setting, g, gprime, pθ, pλ, rangeθ, rangeλ)
    s = setting.s; s0 = setting.s0; X = setting.X; X0 = setting.X0; z = setting.z;
    n = size(X, 1) 
    p = size(X, 2) 
    k = size(X0, 1) 

    function func(θ) 
        Eθ = K(s0, s0, θ, rbf) 
        Σθ = K(s, s, θ, rbf) 
        Bθ = K(s0, s, θ, rbf) 
        choleskyΣθ = cholesky(Σθ) #precompute Cholesky decomposition of Sigma
        choleskyXΣX = cholesky(Hermitian(X'*(choleskyΣθ\X))) #precompute Cholesky decomposition of XSigma\X
        Dθ = Eθ - Bθ*(choleskyΣθ\Bθ') 
        Hθ = X0 - Bθ*(choleskyΣθ\X) 
        Cθ = Dθ + Hθ*(choleskyXΣX\Hθ') 
        args = [Eθ, Σθ, Bθ, choleskyΣθ, choleskyXΣX, Dθ, Hθ, Cθ]
        return args
    end
    function density(λ, z0, args)
        Eθ, Σθ, Bθ, choleskyΣθ, choleskyXΣX, Dθ, Hθ, Cθ = args
        βhat = (X'*(choleskyΣθ\X))\(X'*(choleskyΣθ\g(z, λ))) 
        qtilde = (expr = g(z, λ)-X*βhat; expr'*(choleskyΣθ\expr)) 
        m = Bθ*(cholesky(Σθ)\g(z, λ)) + Hθ*βhat 
        J = λ -> abs(reduce(*, map(x -> gprime(x, λ), z))) 
        post = det(cholesky(Σθ))^(-0.5)*det(choleskyXΣX)^(-0.5)*qtilde[1]^(-(n-p)/2)*J(λ)^(1-p/n)*pθ(θ)*pλ(λ) 
        jac = abs(reduce(*, map(x -> gprime(x, λ), z0)))
        t = LocationScale(m[1], sqrt(qtilde[1]*Cθ[1]/(n-p)), TDist(n-p))
        return post*Distributions.pdf(t, g(z0, λ))*jac, post*Distributions.cdf(t, g(z0, λ))*jac  
    end   
    #simultaneously define PDF and CDF - (reuse integration nodes and weights)
    (z0 ->  int1D(θ -> (args = func(θ); int1D(λ -> density(λ, z0, args)[1], rangeλ)), rangeθ),  
    z0 ->  int1D(θ -> (args = func(θ); int1D(λ -> density(λ, z0, args)[2], rangeλ)), rangeθ))
end


if false

end