
using RecipesBase

@recipe function f(mdl::Model)
    # TODO see RecipesBase.jl, GaussianProcess.jl also has a good example
end

@recipe function f(mdl::Model, s::AbstractVector, x::AbstractVector=s)
    # TODO see RecipesBase.jl
end
