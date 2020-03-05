
import Statistics: median, pdf, cdf

@doc raw"""
"""
function pdf(btg::BTG, x::AbstractVector{R}) where R <: Real
end

@doc raw"""
"""
function cdf(btg::BTG, x::AbstractVector{R}) where R <: Real
end

@doc raw"""
"""
function median(btg::BTG, x::AbstractVector{R}) where R <: Real
    # TODO
end

@doc raw"""
"""
function mode(btg::BTG, x::AbstractVector{R}) where R <: Real
    # TODO
end

@doc raw"""
"""
function credible_interval(
    btg::BTG,
    x::AbstractVector{R};
    mode=:narrow
) where R <: Real
    return credible_interval(btg, x, Val(mode))
end

function credible_interval(
    btg::BTG,
    x::AbstractVector{R},
    ::Val{:equal},
) where R <: Real
    # TODO
end

function credible_interval(
    btg::BTG,
    x::AbstractVector{R},
    ::Val{:narrow},
) where R <: Real
    # TODO
end

@doc raw"""
"""
function map_estimate(btg::BTG)
    # TODO
end

@doc raw"""
"""
function cross_validate(btg::BTG)
    # TODO
end
