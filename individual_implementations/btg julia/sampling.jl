using Distributions
using LinearAlgebra
using Plots

#This file is simply for visualizing the behavior of various kernel functions

function GaussianKernel(x, y, θ::Float64=1.0)
    x = vec(x)
    y = vec(y)
    M = 1/sqrt(2*pi)*exp.(-θ*.5*(x.-y').^2)
    return M    
end 

function generateSample(n, θ)
    x = collect(0:1/n:1-1/n)
    K = GaussianKernel(x, x, θ) + UniformScaling(1e-8) #add diagonal terms to ensure PD-ness
    C = cholesky(K)
    z = C.L*(randn(n))
    return x, z
end

x1, z1 = generateSample(10, 50.0)
x2, z2 = generateSample(50, 50.0)
x3, z3 = generateSample(150, 50.0)
x4, z4 = generateSample(30, 50.0)

display(plot(x1, z1, seriestype = :scatter))
display(plot!(x2, z2, seriestype = :scatter))
display(plot!(x3, z3, seriestype = :scatter))
display(plot!(x4, z4, seriestype = :scatter))

