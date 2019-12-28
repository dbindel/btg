module BTGModel

using Reexport
using SpecialFunctions
using FastGaussQuadrature
using Distributions
using LinearAlgebra

include("./BTGFuncs.jl")

@reexport using .BTGFuncs

export
    Model,
    density,
    distribution

"""
    BTGModel

Settings for the Bayesian Transformed Gaussian model.
"""
struct Model{K<:Correlation,G<:Transform,V<:Covariate}
    k::K
    g::G
    f::V
    X::Array{Float64, 2}
    Y::Array{Float64, 1}
end

function gaussweights(d::Uniform, n)
    x, w = gausslegendre(n)
    return ((d.b - d.a) .* (x .+ 1) ./ 2 .+ d.a), w ./ 2 
end

function gaussweights(d::Normal, n)
    x, w = gausshermite(n)
    return (sqrt(2) * d.σ) .* x .+ d.μ, w ./ sqrt(π)
end

function gaussweights(d::Gamma, n)
    x, w = gausslaguerre(n, d.α - 1)
    return d.θ .* x, w ./ gamma(d.α)
end

function gaussweights(d::Exponential, n)
    x, w = gausslaguerre(n)
    return d.θ .* x, w
end

function weighttensor(f::Parametric, n)
    tups = (gaussweights(getfield(f, s), n) for s in fieldnames(typeof(f)))
    x = collect(Iterators.product((t[1] for t in tups)...))
    w = map(x -> *(x...), Iterators.product((t[2] for t in tups)...))
    return reshape(x, length(x)), reshape(w, length(w))
end



"""
    density(mdl, x, y)

The probability density of the value `y` at the location `x` given a BTG model.

```math
\\mathcal{P}(y_x\\lvert m)
```
"""
function density(mdl::Model, x::Array{Float64, 1}, y::Float64, (kx, kw), (gx, gw))
    n, p = size(mdl.X)
    
    c = 0
    acc = 0

    for i = 1:length(kx)
        θ = kx[i]
        α = kw[i]
        
        cholΣ = cholesky!(Symmetric(kernelmatrix(mdl.k, θ, mdl.X, mdl.X)))
        cholXΣX = cholesky!(Symmetric(mdl.X' * (cholΣ \ mdl.X)))
        B = kernelmatrix(mdl.k, θ, reshape(x, 1, length(x)), mdl.X)
        D = 1.0 - (B * (cholΣ \ B'))[1]
        H = x .- B * (cholΣ \ mdl.X)
        C = D + (H * (cholXΣX \ H'))[1]
        
        for j = 1:length(gx)
            λ = gx[j]
            α′ = gw[j]
            
            Z = mdl.g.(λ, mdl.Y)
            z = mdl.g(λ, y)

            β = cholXΣX \ (mdl.X' * (cholΣ \ Z))
            q = ((Z - mdl.X * β)' * (cholΣ \ (Z - mdl.X * β)))[1]
            J = abs(prod(prime.(Ref(mdl.g), λ, mdl.Y)))
            m = (B * (cholΣ \ Z))[1] + (H * β)[1]

            h = q^(-(n - p) / 2) * J^(1 - p / n) / sqrt(det(cholΣ) * det(cholXΣX))
            T = LocationScale(m, sqrt(q * C), TDist(n - p))
            P = pdf(T, z) * abs(prime(mdl.g, λ, y))
            
            c += α * α′ * h
            acc += α * α′ * h * P   
        end
    end
    
    return acc / c
end

"""
    distribution(mdl, x, y)

The cumulative probability of the value `y` at the location `x` given a BTG model.

```math
\\Phi(y_x\\lvert m)
```
"""
function distribution(mdl::Model, x::Array{Float64, 1}, y::Float64, (kx, kw), (gx, gw))
    n, p = size(mdl.X)
    
    c = 0
    acc = 0

    for i = 1:length(kx)
        θ = kx[i]
        α = kw[i]
        
        cholΣ = cholesky!(Symmetric(kernelmatrix(mdl.k, θ, mdl.X, mdl.X)))
        cholXΣX = cholesky!(Symmetric(mdl.X' * (cholΣ \ mdl.X)))
        B = kernelmatrix(mdl.k, θ, reshape(x, 1, length(x)), mdl.X)
        D = 1.0 - (B * (cholΣ \ B'))[1]
        H = x .- B * (cholΣ \ mdl.X)
        C = D + (H * (cholXΣX \ H'))[1]
        
        for j = 1:length(gx)
            λ = gx[j]
            α′ = gw[j]
            
            Z = mdl.g.(λ, mdl.Y)
            z = mdl.g(λ, y)

            β = cholXΣX \ (mdl.X' * (cholΣ \ Z))
            q = ((Z - mdl.X * β)' * (cholΣ \ (Z - mdl.X * β)))[1]
            J = abs(prod(prime.(Ref(mdl.g), λ, mdl.Y)))
            m = (B * (cholΣ \ Z))[1] + (H * β)[1]

            h = q^(-(n - p) / 2) * J^(1 - p / n) / sqrt(det(cholΣ) * det(cholXΣX))
            T = LocationScale(m, sqrt(q * C), TDist(n - p))
            P = cdf(T, z) * abs(prime(mdl.g, λ, y))
            
            c += α * α′ * h
            acc += α * α′ * h * P   
        end
    end
    
    return acc / c
end

end # module
