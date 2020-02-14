using LinearAlgebra
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
is the number of quadrature nodes (defaults to 10)
"""
function int1D(f, a, b, num="1")
    if num=="1"
        x, w= dm, wm
    elseif num=="2"
        x, w= dm2, wm2
    elseif num=="0"
        x, w = dm0, wm0
    else 
        x, w = dm3, wm3
    end
    int = 0.0
    for i = 1:length(x)
        expr = (b-a)/2*x[i] + (b+a)/2
        int = int + f(expr)*w[i]
    end
    int = (b-a)/2*int
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

"""
n-dimensional Gauss-Legendre integration
"""
function intnD(f, arr, n)

end
