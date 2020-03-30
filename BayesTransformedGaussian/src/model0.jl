include("settings.jl")
include("tensorgrid0.jl")
include("../transforms/transforms.jl")
include("../kernels/kernel.jl")

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
    train::TrainingData #x, Fx, y, p (dimension of each covariate vector), dimension (dimension of each location vector)
    #test::TestingData #s0, X0
    n #number of points in kernel system, if 0 then uninitialized
    g #transform family, e.g. BoxCox()
    k #kernel family, e.g. Gaussian()
    quadtype #Gaussian, Turan, or MonteCarlo
    nodesWeightsλ #integration nodes and weights for λ
    nodesWeightsθ #integration nodes and weights for θ
    θ_buffers #buffers for theta-dependent values
    #Nmax will add this logic later
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

"""
Initialize btg object with training data object (x, Fx, y) and quadtype
"""
function init(train, quadtype = "Gaussian")::BTG
    btg(train, 0, BoxCox(), Gaussian(), quadtype, nothing, nothing, nothing)
end

"""
Sets or resets the quadrature nodes and weights in the btg object.

Notes:
* Number of length scales is encoded in the height of d x 2 matrix rangeθ 
* currently the number of quadrature nodes per dimension is the same

"""
function setrange!(btg::btg, rangeθ, rangeλ; numpts = 12)
     #same length scale for all dimensions or different length scale for all dimensions
    @assert size(rangeθ, 1) == 1 || size(rangeθ, 1) == btg.train.dimension
    @assert size(rangeλ, 1) == 1 
    if btg.quadtype == "MonteCarlo"
        btg.nodesWeightsθ = getMCdata(numpts)
        btg.nodesWeightsλ = getMCdata(numpts)
    else
        if btg.quadtype == "Turan"
            btg.nodesWeightsθ = getTuranQuadratureData(numpts) #use 12 Gauss-Turan integration nodes and weights by default
        elseif btg.quadtype == "Gaussian"
            btg.nodesWeightsθ = getGaussQuadraturedata(numpts)
        elseif
            throw(ArgumentError("Quadrature rule not recognized"))
        end
        btg.nodesWeightsθ = getGaussQuadraturedata(numpts)
        affineTransformNodes(btg.nodesWeightsθ, rangeθ) #will this really mutate btg.nodesWeights?
        #always use Gauss quadrature to marginalize out λ
        btg.nodesWeightsλ = getGaussQuadraturedata(numpts)
        affineTransformNodes(btg.nodesWeightsλ, rangeλ)
    end
    return nothing
end

"""
keep the structure θ_params
update btg.θ_buffers 
"""
function θ_buffers_comp!(btg::BTG)
    # funcθ
end

return btg()
end

function solve(btg::btg)
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