
import Distributions: cdf, pdf
using Distributions
using FastGaussQuadrature
using StaticArrays
using PDMats

mutable struct Model{K<:FunctionPrior{Kernel},
                     G<:FunctionPrior{Transform},
                     Loc<:AbstractMatrix,
                     In<:AbstractMatrix,
                     Out<:AbstractVector}
    k::K # Prior over kernel functiosn
    kn::Int # Number of gauss weights to generate for kernel priors
    g::G # Prior over transforms
    gn::Int # Number of gauss weights to generate for transform priors
    S::Loc # Observed Input data locations
    X::In # Observed Input data covariates
    Y::Out # Observed Output data
end

function cdf(mdl::Model, y::Real; s::AbstractVector, x::AbstractVector=s)
    r = @SVector [0, 0]
end

function pdf(mdl::Model, y::Real; s::AbstractVector, x::AbstractVector=s)
    # TODO
end
