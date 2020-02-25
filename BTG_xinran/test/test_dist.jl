using Distances
using LinearAlgebra
include("../src/kernels/kernel.jl")

#= 
NOT important
compare two ways of compute the kernel matrix
compare my KernelMat function with "Distances.jl"

Result:
    x, y = 100 * rand(1000, 3)
    theta = [rand(), 2 * rand()]
    dist1: 
        346.388 ms (6012 allocations: 137.65 MiB)
    dist2: 
        189.920 ms (3000002 allocations: 328.06 MiB)
    
Conclusion:
    seems like my implementation is faster? 
    so keep it for now.
 =#

function dist1(x, y, theta)
    r = pairwise(Euclidean(), x', y', dims=2)
    kernelfun = r -> Kernel_SE(r, theta)
    K = kernelfun.(r)
    return K
end 

function dist2(x, y, theta)
    K = KernelMat(x, y, Kernel_SE, theta)
    return K
end