module BayesTransformedGaussian

using Distributions, FastGaussQuadrature, Kernel

include("./transforms.jl")
include("./expectation/expectation.jl")
include("./model.jl")
include("./statistics.jl")
include("./plots.jl")

end # module
