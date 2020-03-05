
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

