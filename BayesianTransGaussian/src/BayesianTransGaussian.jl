module BayesianTransGaussian

<<<<<<< HEAD
include("./Model.jl")
using .Model
include("./Funcs.jl")
using .Funcs
include("./Stats.jl")
using .Stats

using Distributions
using SpecialFunctions

"""
    TODO
"""
function load_data()
end

export load_data

# """
# TODO
# """
# struct BTGModel
#     range
#     trendorder
#     gλ
#     kθ
#     priors
# end

# function BTGModel(; range, samplesize, meshsize, trendorder, gλ, kθ, priors)
#     @assert (range[2] - range[1] > 0) "Given range is not positive"
#     @assert (0 > trendorder || trendorder > 2) "Trend order not supported"
#     @assert (kθ <: IsotropicCorrelation) "kθ must be an IsotropicCorrelation type"
#     @assert (gλ <: PowerTransform) "gλ must be a PowerTransform type"
#     for n in collect(Base.Iterators.flatten(fieldnames(kθ), fieldnames(gλ)))
#         @assert (haskey(priors, n)) "No prior found for $(n)"
#     end

#     return BTGModel(range, trendorder, gλ, kθ, priors)
# end

# function train(model, X0, Z0; samplesize, meshsize)
#     @assert (samplesize > 0) "Sample size must be positive"
#     @assert (meshsize > 0) "Mesh size must be positive"
    
#     fX0 = hcat([X0.^i for i in 0:model.trendorder])
    
#     λs = collect(fieldnames(model.gλ))
#     λpriors = [ get(model.priors, λ) for λ in  λs]
#     θs = collect(fieldnames(model.kθ))
#     θpriors = [ get(model.priors, θ) for θ in θs]

#     return ()
# end

# function predict(model, X)
#     # return median, upper bound, lower bound, monte carlo error
# end

# function plot(md, lb, ub, mce)
# end

end # module
