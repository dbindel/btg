
using FastGaussQuadrature
using Distributions

function weights!(w::AbstractMatrix, d::ContinuousUnivariateDistribution, n::Unsigned)
end

function weights(d::Uniform, n::Unsigned)
    # TODO
end

function weights(d::Normal, n::Unsigned)
    # TODO
end

function weights(d::Exponential, n::Unsigned)
    # TODO
end

function weights(d::Gamma, n::Unsigned)
    # TODO
end

function expectation(d:
