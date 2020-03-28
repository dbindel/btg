
"""
BTG object may include
    x: current data
    z: current training values
    n: number of points encorporated in kernel system 
    p: number of covariates 
    g: transform 
    k: kernel type
    quadtype: "Gaussian", "Turan" or "MonteCarlo"
    nodesWeightsλ: stores λ nodes and weights
    nodesWeightsθ: stores θ nodes and weights
    θ_buffers: the old θ_params struct
"""

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
function θ_buffers_comp!(btg)
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

function prediction_comp(btg, WeightTensorGrid)
    # line 100-159 in tensorgrid.jl
    return pdf, cdf, pdf_deriv
end