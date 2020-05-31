using DataFrames
using CSV

function unrate(path)
    df = DataFrame(CSV.File(path*"unrate.csv", header=0))
    #data = convert(Matrix, df[:, 1])
    g = x -> parse(Float64, x)
    target = convert(Array{}, g.(df[2:end, 2]))
    data = collect(1:1:length(target))
    return data, target
end

