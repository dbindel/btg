module BayesianTransGaussian

include("./BTGFuncs.jl")
using .BTGFuncs

export BTGModel

"""
TODO
"""
struct BTGModel{G, K} 
end

function BTGModel(t::Array{R}, z::Array{R};
                  range::Tuple{R,R}, samplesize::Unsigned, meshsize::Unsigned,
                  gλ::PowerTransform{R}, λrange::Tuple{R,R},
                  kθ::IsotropicCorrelation{R}, θrange::Tuple{R,R},
                  trendorder::Unsigned) where {R<:AbstractFloat}
    return BTGModel()
end

end # module
