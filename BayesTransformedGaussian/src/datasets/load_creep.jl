df = DataFrame(CSV.File("../datasets/creeprupt/taka", header=0))
data = convert(Matrix, df[:,3:end])
target = convert(Array, df[:, 2]) 