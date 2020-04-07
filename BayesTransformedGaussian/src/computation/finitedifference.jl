using LinearAlgebra
using Polynomials

include("../tools/plotting.jl")
"""
Use Taylor's Theorem with Remainder to check  
validity of computed Jacobian of f: R^n -> R^m. 

More specifically, check that error of linear approximation decays like O(h^2)

last: smallest power of exp^(-x) we wish to compute in scheme

Requirements: derivative/jacobian must be in the right 2D shape, and not be a 1D list

Hessian is only for functions from R^n -> R^1
"""
function checkDerivative(f, df, x0, hessian = nothing, first = 3, last = 12, num = 10)
    if size(x0, 2) != 1
        x0 = reshape(x0, length(x0), 1) #reshape x0 into 2D column vector or 1D Array
    end
    
    f0 = f(x0)
    df0 = df(x0) 
    n = length(x0)
    m = length(f0) 
    @assert size(f0, 2)==1 #no reason for f0 to be 2D
    println(size(df0))
    println("m: ", m)
    println("n: ", n)
    #@assert size(df0) == (m, n) ||  size(df0) == (m, ) #keep 1D list as possibility
    println("size df0, 1: ", size(df0, 1))
    println("size df0, 2: ", size(df0, 2))
    @assert size(df0, 1) == m && size(df0, 2) == n 
    try 
        df0 = reshape(df0, m, n) #reshapes to m x n if non-scalar
    catch e
    end
    
    if hessian!=nothing 
        @assert m == 1 #we can only compute the Hessian for functions to R^1
        d2f0 = hessian(x0)
        #println(size(d2f0))
        @assert size(d2f0, 1) == n && size(d2f0, 1) ==n 
    end
    #what if x0 or f0 are constants?
    dx = rand(size(x0, 1), size(x0, 2))
    if length(f0) >1
        f0 = reshape(f0, length(f0), 1) #reshape into col vector
    end
    h = collect(first:(last-first)/num:last)
    for i=1:length(h)
        h[i] = exp(-h[i]) 
    end
    A = zeros(length(h))
    for i = 1:length(h) 
        fi = f(x0 .+ h[i]*dx) #dx has the same shape x0, so we can safely do .+, which allows us to add scalar and array
                               #if dx did not have same shape as x0, then .+ could have unintended broadcasting effects
        @assert size(fi, 1) == size(f0, 1) && size(fi, 2) == size(f0, 2) #this lets us do fi .- f0 safely
        if false#debug
            println("x0: ", x0)
            println("dx: ", dx)
            println("fi: ", fi)
            println("f0: ", f0)
            println("df0: ", df0)
            println("increment", h[i]*dx)
            println("h[i]: ", h[i])
        end
            if hessian!=nothing
                inc = h[i] .* dx
                expr1 = fi .- f0
                expr2 = df0 * inc .+ 0.5* inc' * d2f0 * inc
                println("size expr1: ", size(expr1))
                println("size expr2: ", size(expr2))
                @assert size(expr1, 1) == size(expr2, 1) && size(expr1, 2) == size(expr2, 2)
                A[i] = norm(expr1 .- expr2)
                #A[i] = norm((fi - f0) - df0' * inc .- 0.5* inc' * d2f0 * inc)
            else
                expr1 = fi .- f0 
                expr2 = df0 * (h[i] .* dx)
                #println("size expr1: ", size(expr1))
                #println("size expr2: ", size(expr2))
                @assert size(expr1, 1) == size(expr2, 1) && size(expr1, 2)==size(expr2, 2)
                A[i] = norm(expr1 .- expr2)
                #catch DimensionMismatch #we catch the case when f: R^1 -> R^n, in which case  df0'*dx will yield an error
                #    println("Warning: finitedifference dimension mismatch")
                #    @assert size(expr1) == size(expr2)
                #    A[i] = norm(expr1 - expr2)
                #end
        end
    end
    r1 = log.(h)
    r2 = log.(A)
    plt = Plots.plot(r1, r2, title = "Finite Difference Derivative Checker", xlabel = "log of h", ylabel = "log of error")#, fontfamily = font(48, "Courier") , reuse = false)
    return (r1, r2, plt, polyfit(r1, r2, 1))
end
