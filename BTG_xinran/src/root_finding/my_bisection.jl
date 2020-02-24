# find zero for probability density function f(x) 
# where x > 0
# start at x0 
function my_bisection(f, a, b)
    tol = 1e-4
    eps = 1e-5
    kmax = 100
    k = 1
    while k < kmax
        println("ITERATION $k")
        flush(stdout)
        c = (a+b)/2
        if abs(f(c)) < eps || abs(a-b) < tol
            return c
        elseif sign(f(c)) == sign(f(a))
            a = c
            println("Current status: [$a, $b], \n $(f(a)), $(f(b))")
            flush(stdout)
        elseif sign(f(c)) == sign(f(b))
            b = c
            println("Current status: [$a, $b], \n $(f(a)), $(f(b))")
            flush(stdout)
        end
        k += 1
        println("========= FINISHING =========")
        flush(stdout)

    end
    return a, b
end

