using LinearAlgebra
using Distances

function kernel_BTG(r, theta)
    @assert length(theta) == 2
    return theta[1]^(r^(theta[2]))
end

# function kernel_gauss(d, x, y, θ)
#     @assert d == size(x, 1) "input vector x should in dimension d"
#     @assert d == size(y, 1) "input vector y should in dimension d"
#     @assert d == size(θ, 1) "length scale vector should in dimension d"
#     return exp(-dot(θ, (x-y).^2))
# end


function KernelMat(x0, x, kernel, param)
    n0 = size(x0, 1)
    n  = size(x, 1)
    dim = size(x, 2)
    Phi = zeros(n0, n)
    for i in 1:n0, j in 1:n
        if dim == 1
            Phi[i,j] = kernel(norm(x0[i] - x[j]), param)
        else
            Phi[i,j] = kernel(norm(x0[i,:] - x[j,:]), param)
        end
    end
    return Phi
end

