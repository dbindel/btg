

"""
Find root of h(x) in the interval [a, b] using binary search
"""
function bisection(h, a, b, tol=1e-5, nmax=10)
    #find root in [a, b]
    n=1
    c=0
    while n<nmax
        c = (a+b)/2
        if abs(h(c))<tol
            return c
        elseif (sign(h(c))==sign(h(a)))
            a=c
        else
            b=c
        end 
        n=n+1
    end
    return c
end

"""
Computes the symmetric 95% confidence interval about the median of a 
probability distribution given by its PDF using bisection
"""
function confidence(pdf, median, alpha=0.95)
    f = h -> int1D(pdf, median-h, median+h, "2") - alpha
    c = bisection(f, 0, median, 1e-3, 10)
    if abs(c)>median*.9 #this should mean that integration scheme failed to detect spike, indicating that interval is very thin to begin with
        c=.1
    end
    return c
end