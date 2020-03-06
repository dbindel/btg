using Optim

import Statistics: median, pdf, cdf

@doc raw"""
  Add documentation
"""
function pdf(btg::BTG, x::AbstractVector{R}) where R <: Real
    pdf_point_eval = compute_dists!(btg, x)[1]
    return pdf_point_eval
end

@doc raw"""
  Add documentation
"""
function cdf(btg::BTG, x::AbstractVector{R}) where R <: Real
    cdf_point_eval = compute_dists!(btg, x)[2]
end

@doc raw"""
  Add documentation
"""
function median(btg::BTG, x::AbstractVector{R}) where R <: Real
    med = quantile(btg, x)
    return med
end


@doc raw"""
  Add documentation
"""
function quantile(btg::BTG, x::AbstractVector{R}; p::R=.5) where R <: Real
    cdf(x) = cdf(btg, x)
    quantile_func(x) = cdf(x) - p

    function quantile_deriv!(storage, x)
        storage[1] = pdf(btg, x)
    end

    initial_guess = [ rand() ]
    routine = optimize(quantile_func, quantile_deriv!, initial_guess, BFGS())

    quant = Optim.minimizer(routine)

    return quant
end

@doc raw"""
  Add documentation
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
