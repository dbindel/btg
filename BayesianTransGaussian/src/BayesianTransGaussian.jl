module BayesianTransGaussian

using Reexport
using Distributions
using Plots
using Roots
using Distances

include("./BTGModel.jl")

@reexport using .BTGModel

# TODO Finish Configuring Documenter.jl to generate documentation
# TODO When publishing ready, use Registrator.jl
# TODO Optimize code using Profile.jl and BenchmarkTools.jl
# TODO Give Cameron feedback on organization/interface
# TODO Pick which transforms/kernels we want to implement
# TODO Add any covariate transforms we want (polynomial, orthopolynomial for 1d, induced point)
# TODO Add tests to test/runtests.jl, if we want to do unit testing

# See BTGTest.jl for a basic example
# I recommend using Revise.jl (see workflow tips section in Julia docs)

"""
    predictquantile(mdl, q, x)

The inverse cumulative distribution function. The value at location `x` at the `q`th quantile.
"""
function predictquantile(mdl::Model, q, x)
    d = exp.(-pairwise(Euclidean(), mdl.X, reshape(x, 1, length(x)), dims=1))
    startpoint = sum(d .* mdl.Y) / sum(d)
    weights = computeweights(mdl)
    return find_zero(y -> btgdistribution(mdl, x, y, weights) - q, startpoint)
end

"""
    predictmedian(mdl, x)

The value at location `x` at the 0.5th quantile.
"""
predictmedian(mdl::Model, x) = predictquantile(mdl, 0.5, x)

"""
    equalinterval(mdl, density, x)

The equal tailed `p`-credible interval of the value at location `x`.
"""
function equalinterval(mdl::Model, p, x)
    hw = p / 2
    lq = predictquantile(mdl, 0.5 - hw, x)
    m = predictmedian(mdl, x)
    uq = predictquantile(mdl, 0.5 + hw, x)
    return lq, m, uq
end

"""
    predictmode(mdl, x)

The value with the highest probability density at the location `x`.
"""
function predictmode(mdl::Model, x)
    # TODO
    return 1.0
end

"""
    narrowinterval(mdl, p, x)

The narrowest `p`-credible interval of the value at location `x`.
"""
function narrowinterval(mdl::Model, p, x)
    # TODO Need Derivatives for this (?)
    return 0.0, 1.0, 2.0
end

"""
    paramterestimate(mdl)

Gives MAP point estimates of the transformation parameters λ and kernel parameters θ.
"""
function parameterestimate(mdl::Model)
    # TODO Need Derivatives for this
    return (1.0,), (1.0,)
end

# TODO For all plots, consider writing Plots.jl recipes rather than plot functions

"""
    plotbtg(mdl, range, resolution)

For 1D inputs, plots the mode and narrowest intervals at the locations in `range` with
the specified `resolution`.
"""
function plotbtg(mdl::Model, p, range, resolution)
    # TODO
end

"""
    plotdensity(mdl, x, resolution)

Plots the probability density of the values at location `x` with the specified `resolution`.
"""
function plotdensity(mdl::Model, p, x, resolution)
    lq, m, uq = equalinterval(mdl, p, x)
    w = (uq - lq) * 0.1
    r = range(lq - w, stop=uq + w, length=resolution)
    weights = computeweights(mdl)
    plot(y -> btgdensity(mdl, x, y, weights), r) # TODO more output options, proper formatting
    vline!([lq, m, uq])
end

function plotdistribution(mdl::Model, p, x, resolution)
    lq, m, uq = equalinterval(mdl, p, x)
    w = (uq - lq) * 0.1
    r = range(lq - w, stop=uq + w, length=resolution)
    weights = computeweights(mdl)
    plot(y -> btgdistribution(mdl, x, y, weights), r) # TODO more output options, proper formatting
end

"""
    crossvalidate(mdl)

The LOOCV error of given BTGModel.
"""
function crossvalidate(mdl::Model)
    # TODO
    return 1.0
end

# TODO Hyperparameter grid search using crossvalidate?

export
    predictmedian,
    predictmode,
    narrowinterval,
    equalinterval,
    parameterestimate,
    plotbtg,
    plotdensity,
    plotdistribution,
    crossvalidate


end # module
