
import Statistics: median, quantile

function mode(mdl::Model; s::AbstractVector, x::AbstractVector=s)
    # TODO
end

function modeinterval(mdl::Model, p::Real; s::AbstractVector, x::AbstractVector=s)
    # TODO
end

function quantile(mdl::Model, p::Real; s::AbstractVector, x::AbstractVector=s)
    # TODO
end

median(mdl::Model; s::AbstractVector, x::AbstractVector=s) = quantile(mdl, 0.5; s=s, x=x)

function medianinterval(mdl::Model, p::Real; s::AbstractVector, x::AbstractVector=s)
    lb = quantile(mdl, 0.5 - p / 2; s=s, x=x)
    ub = quantile(mdl, 0.5 - p / 2; s=s, x=x)
    return lb, ub
end
