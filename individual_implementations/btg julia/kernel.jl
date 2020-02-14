using Random
using LinearAlgebra

"""
Gaussian/RBF/Squared Exponential correlation function
"""
function rbf(x, y, θ=1.0)
    1/sqrt(2*pi)*exp(-θ*0.5*(norm(x .- y))^2)
end


"""
Builds cross-covariance matrix using corr kernel function.
Ex. RBF, exponential, Matern
 
Inputs are assumed to be 1D arrays or 2D column vectors.
Returns matrix of size len(s1) times len(s2)
"""
function K(s1, s2, θ, corr)  
    K = zeros(size(s1, 1), size(s2, 1))
    for i = 1:size(s1, 1)
        for j = 1:size(s2, 1)
            K[i, j] = corr(s1[i, :], s2[j, :], θ)
        end
    end   
    return size(s1, 1)==size(s2, 1) ? K+UniformScaling(1e-8) : K
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
