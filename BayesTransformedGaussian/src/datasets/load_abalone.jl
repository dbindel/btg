df = DataFrame(CSV.File("../datasets/abalone.csv"), header=0)
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age