using DataFrames
using CSV

df = DataFrame(CSV.File("nodes_weights.csv"))
nodes = convert(Matrix, df[:,1]) #integration nodes for Gauss-Turan Quadrature
weights = convert(Array, df[:, 2:end]) #integration weights 

function eval_Gauss_Turan(f, df, df2, nodes, weights)
    fn = f.(nodes)
    dfn = df.(nodes)
    df2n = df2.(nodes)
    return fn'*weights(:, 1) + dfn'*weights(:, 2)+df2n'*weights(:, 3)
end
