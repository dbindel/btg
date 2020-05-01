using Distances
using LinearAlgebra
include("kernels/kernel.jl")
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
    lower = 0.0005 
    upper = 0.999

    #want k(x, y) to be between .001 and .999
    lb =  2/maxi * ( - log(lower) )
    ub = 2/mini * (- log(upper)) 
    return [min(lb, ub) max(lb, ub)]
end


#test
x = rand(1000, 30)
r = select_single_theta_range(x)
@info "range", r
ker1 = correlation(Gaussian(), r[1], x; jitter = 0, dims=1)
ker2 = correlation(Gaussian(), r[2], x; jitter = 0, dims=1)
@info "min ker 1", minimum(ker1)
@info "max ker 1", maximum(ker1 - UniformScaling(1.0))
@info "min ker 2", minimum(ker2)
@info "max ker 2", maximum(ker2 - UniformScaling(1.0) )


