#this file tests whether using a for loop to apply a function of two variables
# f(x, y) to all pairs of entries of two vectors u, v is faster or if
#if using syntactic sugar f.(X, Y) is faster

g = (x, y)->sin(sqrt(x*y+10000))

x = collect(1:1:400)
y = collect(1:1:400)
yadj = y'

println("for loop time")
@time begin
    z = zeros(400, 400)
    for i = 1:400
        for j = 1:400
            z[i, j] = g(x[i], y[j])            
        end
    end
end
println("syntactic sugar time")
@time w = g.(x, yadj)
10
println(norm(w-z))

#why is broadcasting slower T_T