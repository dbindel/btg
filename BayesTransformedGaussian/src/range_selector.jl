using Distances
using LinearAlgebra
"""
Picks reasonable theta range based on distances between training data points. Used for experimental purposes
"""
function select_single_theta_range(x)
    if size(x, 1) > 8000
        @warn "Forming nxn array of pairwise distances in select_single_theta_range, where n > 8000"
    end
    M = pairwise(SqEuclidean(), x, dims=1)
    maxi = maximum(M)
    @info maxi
    M = M + UniformScaling(maxi) #couldn't think of better way of ignoring zero elts on diagonal
    mini = minimum(M)
    @info mini
    
    #want values in kernel matrix to fall in this range, assuming RBF kernel
    lower = 1e-4
    upper = 0.9999

    #want k(x, y) to be between .001 and .999
    lb =  2/maxi * ( - log(lower) )
    ub = 2/mini * (- log(upper)) 
    return [min(lb, ub) max(lb, ub)]
end

