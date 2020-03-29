include("settings.jl")
include("tensorgrid0.jl")




"""
BTG object may include (some may be unnecessary)
    x: Nmax*d array
    z: Nmax*1 array
    X: Nmax*p array, covariates
    Nmax: maximum number of points BTG could handle
    dim: dimension of the space
    nx: number of points in data
    n: number of points incorporated in kernel system 
    p: number of covariates 
    g: transform 
    k: kernel type
    quadtype: "Gaussian", "Turan" or "MonteCarlo"
    nodesWeightsλ: stores λ nodes and weights
    nodesWeightsθ: stores θ nodes and weights
    θ_buffers: the old θ_params struct
"""

mutable struct btg
    train::TrainingData #s, X, z
    #test::TestingData #s0, X0
    n #number of points
    p #number covariates per point
    g #transform family
    k #kernel family
    quadtype # "Gaussian", "Turan", or "MonteCarlo"
    nodesWeightsλ
    nodesWeightsθ
    θ_buffers
    Nmax #maximum number of points supported
end

"""
Three step of a BTG solver
1. system-building precomputation
    compute or update theta buffers 
2. compute WeightTensorGrid (line 36-99 in tensorgrid.jl)
    W_ij = wt_θ[i] * wt_λ[j] * p(θ_i, λ_j|z) if Gauss quadtype
    W_ij = p(θ_i, λ_j|z)  if Monte Carlo
    W_ijk = ... if Turan
3. prediction computation
    line 100-159 in tensorgrid.jl
"""

function init(train, quadtype, )
    nodesWeightsλ, nodesWeightsθ, θ_buffers
end

return btg()
end

function solve(btg::BTG)
    build_system!(btg) 
    WeightTensorGrid = weight_comp(btg)
    pdf, cdf, pdf_deriv = prediction_comp(btg, WeightTensorGrid)
end

function build_system!(btg::BTG)
    # if currently not points in kernel system
    if btg.n == 0
        new_system!(btg)
        btg.n = size(btg.x, 1)
    else
        for i = (btg.n+1):size(btg.x, 1)
            extend_system!(btg, i)
        end
        btg.n = size(btg.x, 1)
    end
end

"""
set up new system
"""
function new_system!(btg::BTG)
    # get rangeλ and rangeθ
        # fixed and stored in priorInfo or computed from map estimation
    if btg.quadtype == "MonteCarlo"
        btg.nodesWeightsθ = getMCdata()
        btg.nodesWeightsλ = getMCdata()
    else
        if btg.quadtype == "Turan"
            btg.nodesWeightsθ = getTuranQuadratureData() #use 12 Gauss-Turan integration nodes and weights by default
        elseif btg.quadtype == "Gaussian"
            btg.nodesWeightsθ = getGaussQuadraturedata()
        elseif
            throw(ArgumentError("Quadrature rule not recognized"))
        end
        btg.nodesWeightsθ = getGaussQuadraturedata()
        affineTransformNodes(btg.nodesWeightsθ, rangeθ)
        #always use Gauss quadrature to marginalize out λ
        btg.nodesWeightsλ = getGaussQuadraturedata()
        affineTransformNodes(btg.nodesWeightsλ, rangeλ)
    end
    # get prior function
    priorθ = initialize_prior(rangeθ, priortype); 
    priorλ = initialize_prior(rangeλ, priortype); 

    # buffer computation
    btg.gz = btg.g.(btg.z)
    θ_buffers_comp!(btg) 
end

"""
keep the structure θ_params
update btg.θ_buffers 
"""
function θ_buffers_comp!(btg::BTG)
    # funcθ
end

"""
keep the structure θ_params
update btg.θ_buffers 
"""
function extend_system!(btg::BTG, i::Int)
    # extend every buffer of theta with the help of incremental Cholesky    
end


function weight_comp(btg::BTG)
    # line 36-99 in tensorgrid.jl 
    return WeightTensorGrid
end

function prediction_comp(btg::BTG, WeightTensorGrid::Array{Float64})
    # line 100-159 in tensorgrid.jl
    return pdf, cdf, pdf_deriv
end

function add_point!(btg::BTG, x::Array{Float64}, z::Array{Float64,1}, X::Array{Float64})
    btg.nx = size(x, 1)
    btg.x[1:btg.nx, :] = x
    btg.z[1:btg.nx] = z
    btg.X[1:btg.nx, :] = X
    btg.nx += 1
end

function add_points!(btg::BTG, x::Array{Float64}, z::Array{Float64,1}, X::Array{Float64})
    n_new = size(x, 1) # number of new points
    btg.x[btg.nx+1: btg.nx+n_new, :] = x
    btg.z[btg.nx+1: btg.nx+n_new] = z
    btg.X[btg.nx+1: btg.nx+n_new, :] = X
    btg.nx += n_new
end