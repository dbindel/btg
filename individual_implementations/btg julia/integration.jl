using LinearAlgebra
using DataFrames
using CSV
"""
Uses the Golub-Welsch eigenvalue method to compute
Gauss-Legendre quadrature nodes and weights for the domain [-1, 1]

INPUTS: n, the number of desired nodes
OUTPUTS: x and w, the nodes and weights
"""
function gausslegpts(n)
    x = zeros(n-1) 
    for i = 1:n-1
        x[i] = i/sqrt(4*i^2-1)
    end
    x = SymTridiagonal(zeros(n), x) #Jacobi matrix
    d, V = eigen(x)
    w = 2*V[1,:].^2 #factor of 2 comes from the fact that integral of measure is 2
    return d, w
end

#precomute various integration nodes/weights
dm0, wm0 = gausslegpts(24) 
dm, wm = gausslegpts(12) 
dm2, wm2 = gausslegpts(50)
dm3, wm3 = gausslegpts(100)

"""
INPUTS:
f is a function handle, a and b are integration endpoints (a<b), and n 
is the number of quadrature nodes (defaults to 10). Note that f can be
multi-output, in which case int1D integrates each entry individually. This way,
the function lends itself to being composed with Gauss-Turan integration.  
"""
function int1D(f, arr, num="2")
    if num=="1"
        x, w= dm, wm
    elseif num=="2"
        x, w= dm2, wm2
    elseif num=="0"
        x, w = dm0, wm0
    else 
        x, w = dm3, wm3
    end
    sample = f(arr[1])
    res = zeros(size(sample))
    for i = 1:length(x)
        expr = (arr[2]-arr[1])/2 * x[i] .+ (arr[1]+arr[2])/2
        #println(res)
        #println(f(expr) .* w[i])
        res = res .+ f(expr) .* w[i]
    end
    res = (arr[2]-arr[1])/2 .* res
end

function int1D_print(f, arr, num="0")
    if num=="1"
        x, w= dm, wm
    elseif num=="2"
        x, w= dm2, wm2
    elseif num=="0"
        x, w = dm0, wm0
    else 
        x, w = dm3, wm3
    end
    sample = f(arr[1])
    res = zeros(size(sample))
    for i = 1:length(x)
        expr = (arr[2]-arr[1])/2 * x[i] .+ (arr[1]+arr[2])/2
        println(res)
        println(f(expr) .* w[i])
        res = res .+ f(expr) .* w[i]
    end
    res = (arr[2]-arr[1])/2 .* res
end

"""
Integrates function over 2D domain using Gauss-Legendre Quadrature
arr[:, 1] are lower bounds, while arr[:, 2] are upper bounds     
"""
function int2D(f, arr)
    d = size(arr, 1) #number of dimensions
    int = 0.0
    x, w = dm, wm
    avg1 = (arr[1, 2] - arr[1, 1])/2
    avg2 = (arr[2, 2] - arr[2, 1])/2
    for i = 1:length(x)
        for j = 1:length(x)
           expr1 = avg1*x[i] + (arr[1, 2]+arr[1, 1])/2
           expr2 = avg2*x[j] + (arr[2, 2]+arr[2, 1])/2
           int = int + w[i]*w[j]*f(expr1, expr2)
        end
    end
    int = int*avg1*avg2
end

data = DataFrame(CSV.File("nodes_weights_50.csv"))
nodes = convert(Array, data[:,1]) #integration nodes for Gauss-Turan Quadrature
weights = convert(Matrix, data[:, 2:end]) #integration weights 
data2 = DataFrame(CSV.File("nodes_weights_20.csv"))
nodes20  = convert(Array, data2[:,1])
weights20 = convert(Array, data2[:,2:end])

"""
Gauss-Turan integration with 2 derivatives
INPUTS:
arr contains enpoints of integration interval
"""
function Gauss_Turan(F, arr, nodes = nodes, weights=weights)
    a = arr[1]
    b = arr[2]
    nodes = (b-a)/2 .* nodes .+ (b+a)/2
    T = zeros(length(nodes), 3)
    time = @elapsed begin
    for i = 1:length(nodes)
        (T[i, 1], T[i, 2], T[i, 3]) = F(nodes[i])
    end
    end
    println(time)
    fn = T[:, 1]
    dfn = T[:, 2]
    d2fn = T[:, 3]
    return (fn'*weights[:, 1] + dfn'*weights[:, 2] + d2fn'*weights[:, 3])*(b-a)/2
end

"""
n-dimensional Gauss-Legendre integration
"""
function intnD(f, arr, n)

end
