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
    btgdensity,
    btgdistribution,
    computeweights

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
    knumgauss::Int
    gnumgauss::Int
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

function computeweights(mdl::Model)
    kgauss = weighttensor(mdl.k, mdl.knumgauss)
    ggauss = weighttensor(mdl.g, mdl.gnumgauss)
    return (kgauss, ggauss)
end

struct HSSSolver end # TODO Yuanxi Solver

# function Base.size(s::HSSSolver) end # TODO

# function Base.:\(s::HSSSolver, y) end # TODO

# function LinearAlgebra.det(s::HSSSolver) end # TODO

function makesolver!(A)
    return cholesky!(A) # TODO Placeholder
end


# TODO Clean this function up, make it faster
function computedist(mdl::Model, x::Array{Float64, 1}, y::Float64, ((kx, kw), (gx, gw)), f)
    n, p = size(mdl.X)
    
    c = 0
    acc = 0

    for i = 1:length(kx)
        θ = kx[i]
        α = kw[i]
        
        cholΣ = makesolver!(Symmetric(kernelmatrix(mdl.k, θ, mdl.X, mdl.X)))
        cholXΣX = makesolver!(Symmetric(mdl.X' * (cholΣ \ mdl.X)))
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
            P = f(T, θ, λ, z)
            
            c += α * α′ * h
            acc += α * α′ * h * P   
        end
    end
    
    return acc / c
end


"""
    btgdensity(mdl, x, y)

The probability density of the value `y` at the location `x` given a BTG model.

```math
\\mathcal{P}(y_x\\lvert m)
```
"""
function btgdensity(mdl::Model, x::Array{Float64, 1}, y::Float64, weights)
    return computedist(mdl, x, y, weights, function (T, θ, λ, z)
                       return pdf(T, z) * abs(prime(mdl.g, λ, y))
                       end)
end

"""
    btgdistribution(mdl, x, y)

The cumulative probability of the value `y` at the location `x` given a BTG model.

```math
\\Phi(y_x\\lvert m)
```
"""
function btgdistribution(mdl::Model, x::Array{Float64, 1}, y::Float64, weights)
    return computedist(mdl, x, y, weights, (T, _, _, z) -> cdf(T, z))
end

end # module
