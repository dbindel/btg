using LinearAlgebra

# define the basis functions for mean functions
# return matrix X (n by p) such that X_ij = f_j(x_i)
function meanbasisMat(x, p)
    # for now we only consider constant case
    @assert p == 1
    n = size(x, 1)
    X = ones(n)
    return X
end




