module BayesianTransformedGaussian

using Reexport

include("functions/Functions.jl")

@reexport using .Functions

include("Model.jl")
include("Expectation.jl")
include("Statistics.jl")
include("Plotting.jl")

export
    Model

end # module
