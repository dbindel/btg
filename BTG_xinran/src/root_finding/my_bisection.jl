# find zero for probability density function f(x) 
# where x > 0
# start at x0 
function my_bisection(f, a, b)
    tol = 1e-4
    eps = 1e-5
    kmax = 100
    k = 1
    while k < kmax
        c = (a+b)/2
        if abs(f(c)) < eps || abs(a-b) < tol
            return c
        elseif sign(f(c)) == sign(f(a))
            a = c
        elseif sign(f(c)) == sign(f(b))
            b = c
        end
        k += 1
    end
end

