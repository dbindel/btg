module BayesianTransformedGaussian

using Reexport

include("functions/Functions.jl")

@reexport using .Functions

include("Model.jl")
include("Statistics.jl")
include("Plotting.jl")

export
    # Model
    Model,
    # Statistics
    mode,
    modeinterval,
    quantile,
    median,
    medianinterval
    

end # module
