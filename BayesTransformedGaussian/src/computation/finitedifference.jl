include("../tools/plotting.jl")
"""
Use Taylor's Theorem with Remainder to check  
validity of computed derivative. More specifically, check 
that error of linear approximation decays like O(h^2)

last: smallest power of exp^(-x) we wish to compute in scheme
"""
function checkDerivative(f, df, x0, hessian = nothing, first = 3, last = 12, num = 10)
    f0 = f(x0)
    df0 = df(x0) 
    if hessian!=nothing 
        d2f0 = hessian(x0)
    end

    if size(x0, 2)>1
        dx = rand(size(x0, 1), size(x0, 2))
    else
        dx = rand(size(x0, 1))
    end
    h = collect(first:(last-first)/num:last)
    for i=1:length(h)
        h[i] = exp(-h[i]) 
    end
    A = zeros(length(h))
    for i = 1:length(h) 
        fi = f(x0 .+ h[i]*dx)[1] #f incremented
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
                inc = h[i] * dx
                A[i] = norm((fi - f0) - df0' * inc .- 0.5* inc' * d2f0 * inc)
            else
                try
        
                A[i] = norm((fi - f0) .- df0' * (h[i] * dx)) 
                catch DimensionMismatch #we catch the case when f: R^1 -> R^n, in which case  df0'*dx will yield an error
                    println("Warning: finitedifference dimension mismatch")
                    A[i] = norm((fi .- f0) .- df0 .* (h[i] * dx))
                end
        end
    end
    r1 = log.(h)
    r2 = log.(A)
    plt = Plots.plot(r1, r2, title = "Finite Difference Derivative Checker", xlabel = "log of h", ylabel = "log of error")#, fontfamily = font(48, "Courier") , reuse = false)
    return (r1, r2, plt, polyfit(r1, r2, 1))
end
