#installs packages
using Pkg
#stats
Pkg.add("StatsFuns")
Pkg.add("StaticArrays")
Pkg.add("StatsBase") 

#computation
Pkg.add("Polynomials")
Pkg.add("SpecialFunctions")
Pkg.add("PDMats") 
Pkg.add("Distributions") 
Pkg.add("Random") 
Pkg.add("Distances")
Pkg.add("LinearAlgebra")
Pkg.add("FastGaussQuadrature")

#data analysis 
Pkg.add("Plots") 
Pkg.add("MATLAB")
Pkg.add("DataFrames")
Pkg.add("CSV") 

#timing
Pkg.add("TimerOutputs")

#optimization
Pkg.add("Cubature")
Pkg.add("Optim")

#visualization
Pkg.add("PyPlot")