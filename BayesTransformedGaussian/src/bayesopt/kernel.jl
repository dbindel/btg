using Random
using LinearAlgebra
using Distances

"""
Gaussian/RBF/Squared Exponential correlation function
"""
function rbf(x, y, θ=1.0)
    1/sqrt(2*pi)*exp.(-θ*0.5*(norm(x .- y))^2)
end

"""
Gaussian/RBF/Squared Exponential correlation function
"""
function rbf_single(dist, θ=1.0)
    1/sqrt(2*pi)*exp.(-θ*0.5*(dist)^2)
end

"""
Derivative of Gaussian/RBF/Squared Exponential correlation function
with respect to hyperparameter theta
"""
function rbf_prime(x, y, θ=1.0)
    - 1/sqrt(2*pi) * 0.5 * norm(x.-y)^2 * exp.(-θ*0.5*(norm(x .- y))^2)
end

"""
Derivative of Gaussian/RBF/Squared Exponential correlation function
with respect to hyperparameter theta
"""
function rbf_prime_single(dist, θ=1.0)
    - 1/sqrt(2*pi) * 0.5 * dist^2 * exp.(-θ*0.5*(dist)^2)
end

"""
Second derivative of Gaussian/RBF/Squared Exponential correlation function
with respect to hyperparameter theta
"""
function rbf_prime2(x, y, θ=1.0)
    1/sqrt(2*pi) * 0.25 * norm(x.-y)^4 * exp.(-θ*0.5*(norm(x .- y))^2)
end

"""
Second derivative of Gaussian/RBF/Squared Exponential correlation function
with respect to hyperparameter theta
"""
function rbf_prime2_single(dist, θ=1.0)
    1/sqrt(2*pi) * 0.25 * dist^4 * exp.(-θ*0.5*(dist)^2)
end

"""
Builds cross-covariance matrix using corr kernel function.
Ex. RBF, exponential, Matern
 
Inputs are assumed to be 1D arrays or 2D column vectors.
Returns matrix of size len(s1) times len(s2)
"""
function K(s1, s2, θ, corr=rbf)  
    K = zeros(size(s1, 1), size(s2, 1))
    if s1 != s2
        for i = 1:size(s1, 1)
            for j = 1:size(s2, 1)
                cur = corr(s1[i, :], s2[j, :], θ)
                K[i, j] = isa(cur, Array) ? cur[1] : cur 
            end
        end   
    else #s1==s2 can compute half the matrix and fill in other half
        for i = 1:size(s1, 1)
            for j = 1:i
                cur = corr(s1[i, :], s2[j, :], θ)
                K[i, j] = K[j, i] = isa(cur, Array) ? cur[1] : cur 
            end
        end  
    end
    return size(s1, 1)==size(s2, 1) ? K+UniformScaling(1e-8) : K
end

#rows of s1 and s2 contain the coordinates of each location
function fastK(s1, s2, θ, corr=rbf_single)
    corrtheta = dist -> corr(dist, θ)
    res = corrtheta.(pairwise(Euclidean(), s1, s2, dims=1))
    return size(s1, 1)==size(s2, 1) ? res + UniformScaling(1e-8) : res
end

"""
Samples from a Normal distribution with mean 0 and covariance matrix K
Scales result by 40. Works in 1D.
"""
function sample(K)
    c = cholesky(K)
    v = 40*randn(size(K, 1), 1)
    return c.L*(v)
end
