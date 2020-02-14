
using DataFrames
using CSV

include("btgDerivatives.jl")
include("legpts.jl")    
using Polynomials

using .btgDeriv

data = DataFrame(CSV.File("nodes_weights_50.csv"))
nodes = convert(Array, data[:,1]) #integration nodes for Gauss-Turan Quadrature
weights = convert(Matrix, data[:, 2:end]) #integration weights 

#f = x -> sin(x^2)
#df = x -> 2*x*cos(x^2)
#df2 = x -> 2*cos(x^2)-4*x^2*sin(x^2)

#f = x -> x^14
#df = x-> 14 * x^13
#df2 = x -> 182 * x^12

#f =  x -> x^70+
#df = x -> 70* x^69
#df2 = x -> 4830*x^68
#test
#f = x -> sin.(exp.(3 .-x))
#df = x -> -exp.(3 .-x) .* cos.(exp.(3 .-x))
#df2 = x -> exp.(3 .-2*x) .*(exp.(x) .* cos.(exp.(3 .-x)).-exp(3) .* sin.(exp.(3 .-x)))

(h, A) = checkDerivative(df, df2, .5)
println(polyfit(h, A, 1))

function eval_Gauss_Turan(f, df, df2, nodes, weights)
    fn = f.(nodes)
    dfn = df.(nodes)
    df2n = df2.(nodes)
    return fn'*weights[:, 1] + dfn'*weights[:, 2] + df2n'*weights[:, 3]
end

res1 = int1D(f, -1, 1, "3")
println("Gauss error: ", abs(res1-0.080873396420834301832))

res = eval_Gauss_Turan(f, df, df2, nodes, weights)
println("Turan error: ", abs(res - 0.080873396420834301832))
