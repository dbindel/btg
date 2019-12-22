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
    Covariate,
    Identity,
    kernelmatrix,
    getparam,
    sampleparam,
    prime

export
    BTGModel,
    predictmedian,
    predictmode,
    narrowinterval,
    equalinterval,
    parameterestimate,
    plotbtg,
    crossvalidate

"""
    BTGModel

Settings for the Bayesian Transformed Gaussian model.
"""
struct BTGModel{K<:Correlation,G<:Transform,V<:Covariate}
    k::K
    g::G
    f::V
    X::Array{Float64, 2}
    Y::Array{Float64, 1}
end

"""
    density(mdl, x, y)

The probability density of the value `y` at the location `x` given a BTG model.

```math
\\mathcal{P}(y_x\\lvert m)
```
"""
function density(mdl::BTGModel, x, y)
    # TODO
    return 1.0
end

"""
    distribution(mdl, x, y)

The cumulative probability of the value `y` at the location `x` given a BTG model.

```math
\\Phi(y_x\\lvert m)
```
"""
function distribution(mdl::BTGModel, x, y)
    # TODO
    return 1.0
end

"""
    predictquantile(mdl, q, x)

The inverse cumulative distribution function. The value at location `x` at the `q`th quantile.
"""
function predictquantile(mdl::BTGModel, q, x)
    # TODO
    return 1.0
end

"""
    predictmedian(mdl, x)

The value at location `x` at the 0.5th quantile.
"""
predictmedian(mdl::BTGModel, x) = quantile(mdl, 0.5, x)

"""
    equalinterval(mdl, density, x)

The equal tailed `p`-credible interval of the value at location `x`.
"""
function equalinterval(mdl::BTGModel, p, x)
    hw = p / 2
    return quantile(mdl, 0.5 - hw, x), quantile(mdl, 0.5, x), quantile(mdl, 0.5 + hw, x)
end

"""
    predictmode(mdl, x)

The value with the highest probability density at the location `x`.
"""
function predictmode(mdl::BTGModel, x)
    # TODO
    return 1.0
end

"""
    narrowinterval(mdl, p, x)

The narrowest `p`-credible interval of the value at location `x`.
"""
function narrowinterval(mdl::BTGModel, p, x)
    # TODO
    return 0.0, 1.0, 2.0
end

"""
    paramterestimate(mdl)

Gives MAP point estimates of the transformation parameters λ and kernel parameters θ.
"""
function parameterestimate(mdl::BTGModel)
    # TODO
    return (1.0,), (1.0,)
end

"""
    plotbtg(mdl, range, resolution)

For 1D inputs, plots the mode and narrowest intervals at the locations in `range` with
the specified `resolution`.
"""
function plotbtg(mdl::BTGModel, range, resolution)
    # TODO
end

"""
    plotdensity(mdl, x, resolution)

Plots the probability density of the values at location `x` with the specified `resolution`.
"""
function plotdensity(mdl::BTGModel, x, resolution)
    # TODO
end

"""
    crossvalidate(mdl)

The LOOCV error of given BTGModel.
"""
function crossvalidate(mdl::BTGModel)
    # TODO
    return 1.0
end

end # module
