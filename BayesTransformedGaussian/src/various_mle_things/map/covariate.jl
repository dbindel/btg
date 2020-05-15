abstract type AbstractCovariate end

struct Linear <: AbstractCovariate end
(::Linear)(Y) = [ones(1, size(Y, 2)); Y]
