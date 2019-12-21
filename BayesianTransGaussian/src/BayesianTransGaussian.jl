module BayesianTransGaussian

using Reexport
using Distributions

include("./BTGFuncs.jl")

using .BTGFuncs

export
    Correlation,
    IsotropicCorrelation,
    SquaredExponential,
    Transform,
    BoxCox,
    kernelmatrix,
    getparam,
    sampleparam,
    prime

"""
    GaussQuadrature(n)

Settings for a gauss quadrature rule.
"""
struct GaussQuadrature
    n::UInt
end

"""
    MonteCarloQuadrature(n)

Settings for a monte carlo quadrature
"""
struct MonteCarloQuadrature
    n::UInt
end

"""
    BTGModel
"""
struct BTGModel where {K::Correlation, G::Transform}
    k::Correlation
    g::Transform
end

export
    BTGModel,
    GaussQuadrature,
    MonteCarloQuadrature

end # module
