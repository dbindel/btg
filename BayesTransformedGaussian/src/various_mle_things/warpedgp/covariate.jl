abstract type AbstractCovariate end

struct Linear <: AbstractCovariate end
covariate(::Linear, X) = [ones(1, size(X, 2)); X]

struct Constant <: AbstractCovariate end
covariate(::Constant, X) = ones(eltype(X), 1, size(X, 2))
