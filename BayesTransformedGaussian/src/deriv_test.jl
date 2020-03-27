include("computation/derivatives.jl")
using Plots

f = x-> x[1]*x[2] + x[2]^3 #derivative
df = x -> [x[2];  x[1]+3*x[2]^2] #Jacobian, pass in as column vector
d2f = x ->  [0 1; 1 6*x[2]] #Hessian

(h, A) = checkDerivative(f, df, [1.0;3.0], d2f, 3, 6)
polyfit(h, A, 1)
plt1 = Plots.plot(h, A, title = "Finite Difference Derivative Checker", xlabel = "log of h", ylabel = "log of error",fontfamily=font(48, "Courier") , reuse = false)
#plot(polyfit(h, A, 1), reuse = true)
println("derivative of p(z0|z)")
println(polyfit(h, A, 1))  
display(plt1)