module BayesianTransGaussian

using Reexport
using Distributions

include("./BTGModel.jl")

@reexport using .BTGModel


"""
    predictquantile(mdl, q, x)

The inverse cumulative distribution function. The value at location `x` at the `q`th quantile.
"""
function predictquantile(mdl::Model, q, x)
    # TODO
    return 1.0
end

"""
    predictmedian(mdl, x)

The value at location `x` at the 0.5th quantile.
"""
predictmedian(mdl::Model, x) = quantile(mdl, 0.5, x)

"""
    equalinterval(mdl, density, x)

The equal tailed `p`-credible interval of the value at location `x`.
"""
function equalinterval(mdl::Model, p, x)
    hw = p / 2
    return quantile(mdl, 0.5 - hw, x), quantile(mdl, 0.5, x), quantile(mdl, 0.5 + hw, x)
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
    # TODO
    return 0.0, 1.0, 2.0
end

"""
    paramterestimate(mdl)

Gives MAP point estimates of the transformation parameters λ and kernel parameters θ.
"""
function parameterestimate(mdl::Model)
    # TODO
    return (1.0,), (1.0,)
end

"""
    plotbtg(mdl, range, resolution)

For 1D inputs, plots the mode and narrowest intervals at the locations in `range` with
the specified `resolution`.
"""
function plotbtg(mdl::Model, range, resolution)
    # TODO
end

"""
    plotdensity(mdl, x, resolution)

Plots the probability density of the values at location `x` with the specified `resolution`.
"""
function plotdensity(mdl::Model, x, resolution)
    # TODO
end

"""
    crossvalidate(mdl)

The LOOCV error of given BTGModel.
"""
function crossvalidate(mdl::Model)
    # TODO
    return 1.0
end

export
    predictmedian,
    predictmode,
    narrowinterval,
    equalinterval,
    parameterestimate,
    plotbtg,
    crossvalidate


end # module
