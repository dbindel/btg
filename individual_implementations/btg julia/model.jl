using LinearAlgebra
using Distributions
using SpecialFunctions
using PDMats
using Printf
include("kernel.jl")

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
function z0 -> p(z_0|z) 
"""
function model(X, X0, s, s0, g, gprime, pθ, pλ, z, rangeθ, rangeλ)
    """
    Computes p(theta, lambda | z)*p(z_0| theta, lambda, z)
    """
    function density(θ, λ, z0) 
        n = size(X, 1) # number of training points
        p = size(X, 2) # number of covariates
        k = size(X0, 1) # number of prediction locations
        Eθ = K(s0, s0, θ, rbf) #test-test covariance matrix
        Σθ = K(s, s, θ, rbf) #train-train covariance matrix
        Bθ = K(s0, s, θ, rbf) #test-train cross covariance matrix
        βhat = (X'*(Σθ\X))\(X'*(Σθ\g(z, λ))) #beta hat: regressors for mean function/solution to least squares problem
        qtilde = (expr = g(z, λ)-X*βhat; expr'*(Σθ\expr)) #qtilde 
        Dθ = Eθ - Bθ*(Σθ\Bθ') #D_theta 
        Hθ = X0 - Bθ*(Σθ\X) #H_theta
        m = Bθ*(Σθ\g(z, λ)) + Hθ*βhat #m_{theta, lambda}
        Cθ = Dθ + Hθ*((X'*(Σθ\X))\Hθ') #C_theta
        J = λ -> abs(reduce(*, map(x -> gprime(x, λ), z))) #Jacobian function
        post = det(Σθ)^(-0.5)*det(X'*(Σθ\X))^(-0.5)*qtilde[1]^(-(n-p)/2)*J(λ)^(1-p/n)*pθ(θ)*pλ(λ) # p(theta, lambda | z) (Eq. 8) 
        jac = abs(reduce(*, map(x -> gprime(x, λ), z0)))
        C = PDMat(qtilde[1]*Cθ); #convert to PDMat to feed into multivariate Student's t-distribution
        t = MvTDist(n-p, vec(m), C/(n-p)); #T-distribution with n-p df, mean m, and covariance C/(n-p) 
        like = Distributions.pdf(t,[g(z0, λ)])*jac # p(z_0| theta, lambda, z) as in Equation 12
        return post*like
    end
    z0 -> int2D((θ, λ) -> density(θ, λ, z0), vcat(rangeθ, rangeλ)) 
end

    