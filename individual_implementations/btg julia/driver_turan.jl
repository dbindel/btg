
using DataFrames
using CSV

include("btgDerivatives.jl")
include("integration.jl")    
using Polynomials

#f = x -> sin(x^2)
#df = x -> 2*x*cos(x^2)
#df2 = x -> 2*cos(x^2)-4*x^2*sin(x^2)

#f = x -> x^14
#df = x-> 14 * x^13
#df2 = x -> 182 * x^12

f =  x -> x .^47 + x.^24
df = x -> 47 * x .^46 + 24*x.^23
df2 = x -> 2162*x .^45 + 552*x.^22

#f = x -> sin.(exp.(3 .-x))
#df = x -> -exp.(3 .-x) .* cos.(exp.(3 .-x))
#df2 = x -> exp.(3 .-2*x) .*(exp.(x) .* cos.(exp.(3 .-x)).-exp(3) .* sin.(exp.(3 .-x)))

#f = x-> sin.(x .+pi/2)
#df = x-> cos.(x .+pi/2)
#df2 = x -> -sin.(x .+pi/2)

(nodes, weights) = getTuranData()

(h, A) = checkDerivative(df, df2, .5)
println(polyfit(h, A, 1))

function eval_Gauss_Turan(f, df, df2, nodes, weights)
    fn = f.(nodes)
    dfn = df.(nodes)
    df2n = df2.(nodes)
    return fn'*weights[:, 1] + dfn'*weights[:, 2] + df2n'*weights[:, 3]
end

res1 = int1D(f, [-1, 1.1])
println("Gauss error: ", abs(res1 - 2.47375))

F = x -> [f(x); df(x); df2(x)]
res = Gauss_Turan(F, [-1, 1.1], nodes, weights)
println("Turan error: ", abs(res - 2.47375))

