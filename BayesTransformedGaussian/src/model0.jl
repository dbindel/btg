include("settings.jl")
include("quadrature/tensorgrid0.jl")
include("transforms/transforms.jl")
include("kernels/kernel.jl")
include("priors/priors.jl")
include("computation/buffers.jl")

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
    train_data::AbstractTrainingData #x, Fx, y, p (dimension of each covariate vector), dimension (dimension of each location vector)
    n::Int64 #number of points in kernel system, if 0 then uninitialized
    g:: NonlinearTransform #transform family, e.g. BoxCox()
    k::AbstractCorrelation  #kernel family, e.g. Gaussian()
    prior::priorType #prior type, e.g. uniform
    quadType::String #Gaussian, Turan, or MonteCarlo
    nodesWeightsθ #integration nodes and weights for θ
    nodesWeightsλ #integration nodes and weights for λ; nodes and weights should remain constant throughout the lifetime of the btg object
    train_buffer_dict::Dict{Float64, train_buffer}  #buffer for each theta value
    test_buffer_dict::Dict{Float64, test_buffer} #buffer for each theta value
    #buffers for theta-dependent values
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
Initialize btg object
NOTE:
- rangeθ has dimension 2, its rows represent integration ranges for each length scale parameter
"""
function init(train_data::AbstractTrainingData, rangeθ, rangeλ; corr::AbstractCorrelation = Gaussian(), prior::priorType = Uniform() , quadtype::String = "Gaussian", transform:NonlinearTransform = BoxCox())::btg
    #a btg object really should contain a bunch of train buffers correpsonding to different theta-values
    #we should add some fields to the nodesweights_theta data structure to figure out the number of dimensions we are integrating over...should we allow different length scale ranges w/ different quadrature nodes? I think so??
    nodesWeightsθ = nodesWeights(rangeθ, quadtype)
    nodesWeightsλ = nodesWeights(rangeλ, quadtype)
    train_buffer_dict  = init_train_buffer_dict(nodesWeightsθ, training_dataa, corr)
    test_buffer_dict = init_test_buffer_dict()
    btg(train_data, 0, transform, corr, prior, quadtype, nodesWeightsθ, nodsesWeightsλ, train_buffer_dict, nothing)
end

#workflow is:
#1) set_test_data
#2) solve, i.e. get pdf and cdf 
#3) update_system if needed

#x, Fx, y

function update_system!(btg::btg, x0, Fx0, y0)
    update!(btg.train_data, x0, Fx0, y0)    
    update_train_buffer!(btg.train_buffer, btg.train)
    update_test_buffer!(btg.train_buffer, btg.test_buffer, btg.trainingData)
end


function solve(btg::btg)
    
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
    # get prior function
    priorθ = initialize_prior(rangeθ, priortype); 
    priorλ = initialize_prior(rangeλ, priortype); 

    # buffer computation
    btg.gz = btg.g.(btg.z)
    θ_buffers_comp!(btg) 
end


function weight_comp(btg::BTG)#depends on train_data and not test_data
    # line 36-99 in tensorgrid.jl 
    return WeightTensorGrid
end

function prediction_comp(btg::BTG, WeightTensorGrid::Array{Float64}) #depends on both train_data and test_data
    update_test_buffer!(train_buffer::train_buffer, test_buffer::test_buffer, trainingData::AbstractTrainingData, testingData::AbstractTrainingData)
    # line 100-159 in tensorgrid.jl
    return pdf, cdf, pdf_deriv
end

