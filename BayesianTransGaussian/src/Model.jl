"""
    TODO
"""
module Model

export BTGModel, BTGInitializer, predict

"""
    BTGModel

A struct containing parameter settings defining a Bayesian Transformed Gaussian Model.
"""
struct BTGModel
end

"""
    BTGData

A struct containing the parameter dependent data needed for one sample 
from the model.
"""
struct BTGData
    gλ # transformation function
    kθ # kernel function
    cholΣ # Cholesky factor of the correlation matrix
    R  # QR = cholΣX
    gZ # transformed data
    βh # Estimated covariate weights
    q # Covariance vector for g(Z0) ∼ N(μ, q)
    pz # p(z | λ, θ)
    M # Location parameter for posterior Student T
    qC # Scale parameter
end

"""
    init()

Takes in parameter settings for a BTG model and outputs a function
which will fit a BTGModel according to those parameters.
"""
function init()
    function fit(X0, Z0)
    end

    return fit
end

"""
    predict(model, X)

Takes a BTGModel and a matrix X and predicts the corresponding Z vector conditioned
on X0 and Z0. 
"""
function predict(model, X)
end

"""
    compute_βh()

Compute the regression coefficients conditioned on X0, Z0 and a particular draw of λ,θ
"""
function compute_βh() # Yuanxi - Solves with covariance matrix used here
end

"""
    compute_q

Compute the covariance for gaussian process gλ(Z0) ∼ N(μ, q)
"""
function compute_q() # Yuanxi - Solves with covariance matrix used here
end

"""
    compute_jacobian()

Compute the jacobian of the transformation function gλ applied to Z0
"""
function compute_jacobian()
end

"""
    compute_location()

Compute the location parameter for the posterior student T distribution describing
Z.
"""
function compute_location() # Yuanxi - Solves with covariance matrix used here
end

"""
    compute_scale()

Compute the scale parameter for the posterior student T distribution describing Z.
"""
function compute_scale() # Yuanxi - Solves with covariance matrix used here
end

"""
    parameter_pdf()

Compute an estimate of p(λ, θ | z0).
"""
function parameter_pdf() # Relevant to Darian and Leo - need derivatives of this to optimise
end

"""
    prediction_pdf()

Compute an estimate of p(z | λ, θ, z0)
"""
function prediction_pdf()
end

"""
    observed_pdf()

Compute an estimate of p(z0 | λ, θ)
"""
function observed_pdf()
end

end # module
