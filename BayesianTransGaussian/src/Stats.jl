"""
    Stats

Contains a variety for functions to calculate a variety of statistical quantities pertinent to 
the evaluation fo the BTGModel.
"""
module Stats

include("./Model.jl")
using .Model

export
    stats,
    estimate_CDF,
    estimate_quantile,
    symmetric_CI,
    estimate_parameters,
    error_estimate,
    validation_error

"""
    stats()

Returns a tuple containing the median and 95% prediction interval.
"""
function stats()
end

"""
    estimate_CDF()

Gives an estimate of F(z | z0) = P(Z <= z | z0)
"""
function estimate_CDF()
end

"""
    estimate_quantile()

Gives an estimate of invF(z | z0)
"""
function estimate_quantile()
end

"""
    symmetric_CI()

Computes a symmetric confidence interval about the median st P(lb <= md <= ub) = α
"""
function symmetric_CI()
end

"""
    estimate_parameters()
"""
function estimate_parameters() # Darian & Leo - Derivatives will be used here to compute MAP estimates
end

"""
    error_estimate()

Compute the estimate ϕ of the maximum estimated standard error in the monte
carlo estimation of p(z | z0)
"""
function error_estimate()
end

"""
    validation_error()

Compute the Leave One Out cross validation error of the model.
"""
function validation_error() # See 6210 course notes.
end

end
