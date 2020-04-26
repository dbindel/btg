df = DataFrame(CSV.File("../datasets/Ailerons/ailerons.data", header=0))
data = convert(Matrix, df[:,1:40])
target = convert(Array, df[:, 41]) 