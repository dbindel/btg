#include("transforms/transforms.jl")
#include("kernels/kernel.jl")
#include("priors/priors.jl")
#include("computation/buffers0.jl")
#include("dataStructs.jl")
using StatsFuns

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
    trainingData::AbstractTrainingData #x, Fx, y, p (dimension of each covariate vector), dimension (dimension of each location vector)
    testingData::AbstractTestingData 
    n::Int64 #number of points in kernel system, if 0 then uninitialized
    g:: NonlinearTransform #transform family, e.g. BoxCox()
    k::AbstractCorrelation  #kernel family, e.g. Gaussian()
    quadType::Array{String,1} #type for theta and lambda respectively, Gaussian, Turan, or MonteCarlo
    priorθ::priorType
    priorλ::priorType
    nodesWeightsθ #integration nodes and weights for θ
    nodesWeightsλ #integration nodes and weights for λ; nodes and weights should remain constant throughout the lifetime of the btg object
    θλbuffer_dict::Dict{Any, θλbuffer} #key should be theta-lambda pair
    train_buffer_dict::Dict{Union{Array{T, 1}, T} where T<: Real, train_buffer}   #buffer for each theta value
    test_buffer_dict::Dict{Union{Array{T, 1}, T} where T<: Real, test_buffer}  #buffer for each theta value
    capacity::Int64
    function btg(trainingData::AbstractTrainingData, rangeθ, rangeλ; corr = Gaussian(), priorθ = Uniform(rangeθ), priorλ = Uniform(rangeλ), quadtype = ["Gaussian", "Gaussian"], transform = BoxCox())
        @assert typeof(corr)<:AbstractCorrelation
        @assert typeof(priorθ)<:priorType
        @assert typeof(priorλ)<:priorType
        @assert typeof(quadtype)<:Array{String,1}
        @assert typeof(transform)<: NonlinearTransform
        @assert Base.size(rangeθ, 1) == getDimension(trainingData) || Base.size(rangeθ, 1)==1
        #a btg object really should contain a bunch of train buffers correpsonding to different theta-values
        #we should add some fields to the nodesweights_theta data structure to figure out the number of dimensions we are integrating over...should we allow different length scale ranges w/ different quadrature nodes? I think so??
        nodesWeightsθ = nodesWeights(rangeθ, quadtype[1]; num_pts = 18, num_MC = 400)
        nodesWeightsλ = nodesWeights(rangeλ, quadtype[2]; num_pts = 18, num_MC = 400)
        θλbuffer_dict = Dict{Any, θλbuffer}
        train_buffer_dict  = init_train_buffer_dict(nodesWeightsθ, trainingData, corr, quadtype[1])
        test_buffer_dict = Dict{Union{Array{T, 1}, T} where T<:Real, test_buffer}(arr => test_buffer() for arr in keys(train_buffer_dict)) #initialize keys of dict with unitialized test buffer values
        cap = getCapacity(trainingData)
        new(trainingData, testingData(), 0, transform, corr, quadtype, priorθ, priorλ, nodesWeightsθ, nodesWeightsλ, train_buffer_dict, test_buffer_dict, θλbuffer_dict, cap)
    end
end

#workflow is:
#1) set_test_data
#2) solve, i.e. get pdf and cdf 
#3) update_system if needed
#x, Fx, y

"""
Updates btg object with newly observed data points. Used in the context of BO.
"""
function update!(btg::btg, x0, Fx0, y0) #extend system step, invariant is 
    update!(btg.trainingData, x0, Fx0, y0)    
    update_train_buffer!(btg.train_buffer, btg.train)
end

"""
"""
function solve(btg::btg; validate = 0)
    weightTensorGrid = weight_comp(btg; validate = validate)
    (pdf, cdf, dpdf, quantInfo) = prediction_comp(btg, weightTensorGrid; validate = validate)
end

"""
Stably compute weights in the mixture of T-distributions, i.e. |Σθ|^(-1/2) * |X'ΣθX|^(-1/2) * qtilde^(-(n-p)/2) * Jac(z)^(1-p/n) * pθ(θ) * pλ(λ),
for all combinations of quadrature nodes in θ and λ
"""
function weight_comp(btg::btg; validate = 0)#depends on train_data and not test_data
    nt1 = btg.nodesWeightsθ.d   #number of dimensions of theta 
    nt2 = btg.nodesWeightsθ.num #number of theta quadrature in each dimension
    nl1 = btg.nodesWeightsλ.d   #number of dimensions of lambda 
    nl2 = btg.nodesWeightsλ.num #number of lambda quadrature in each dimension
    n =  btg.trainingData.n; p = btg.trainingData.p #number of training points and dimension of covariates
    Fx = getCovariates(btg.trainingData)
    z = getLabel(btg.trainingData)
    g = (x, λ) -> btg.g(x, λ); dg = (x, λ) -> partialx(btg.g, x, λ); lmbda = λ -> g(z, λ)
    gλvals = Array{Float64, 2}(undef, length(z), nl2) #preallocate space to store gλz arrays
    for i = 1:nl2
        gλvals[:, i] = lmbda(btg.nodesWeightsλ.nodes[i])
    end
    jacvals = zeros(1, nl2)  #compute exponents of Jac(z)^(1-p/n)
    for i = 1:nl2
        jacvals[i] = sum(log.(abs.(map( x-> dg(x, btg.nodesWeightsλ.nodes[i]), z))))
    end
    
    if endswith(btg.quadType[1], "MonteCarlo") && endswith(btg.quadType[2], "MonteCarlo")
        weightsTensorGrid = Array{Float64, 1}(undef, nt2) 
        R = CartesianIndices(weightsTensorGrid)
    elseif btg.quadType == ["Gaussian", "Gaussian"]
        weightsTensorGrid = Array{Float64, nt1+nl1}(undef, Tuple(vcat([nt2 for i = 1:nt1], [nl2 for i = 1:nl1]))) #initialize tensor grid
        R = CartesianIndices(weightsTensorGrid)
        for I in R #I is multi-index
            weightsTensorGrid[I] = getProd(btg.nodesWeightsθ.weights, btg.nodesWeightsλ.weights, I) #this step can be simplified because the tensor is symmetric (weights are the same along each dimension)
        end
    else
        weightsTensorGrid = Array{Float64, 2}(undef, nt2, nl2) 
        R = CartesianIndices(weightsTensorGrid)
        weightsTensorGrid = repeat(btg.nodesWeightsλ.weights, nt2, 1) # add lambda weights 
    end

    powerGrid = similar(weightsTensorGrid) 
    for I in R
        r1 = (endswith(btg.quadType[1], "MonteCarlo") && endswith(btg.quadType[2], "MonteCarlo")) ? I : Tuple(I)[1:end-1] 
        r2 = Tuple(I)[end]
        t1 = btg.quadType[1] == "Gaussian" ? getNodeSequence(getNodes(btg.nodesWeightsθ), r1) : getNodes(btg.nodesWeightsθ)[:, r1[1]]
        t2 = getNodes(btg.nodesWeightsλ)[:, r2] 
        train_buffer = btg.train_buffer_dict[t1] #look up train buffer based on combination of theta quadrature nodes
        
        choleskyXΣX = train_buffer.choleskyXΣX
        choleskyΣθ = train_buffer.choleskyΣθ
        gλz = gλvals[:, r2]
        βhat = choleskyXΣX\(Fx'*(choleskyΣθ\gλz)) 
        qtilde = (expr = gλz-Fx*βhat; expr'*(choleskyΣθ\expr))

        qTensorGrid = -(n-p)/2 * log(qtilde)  #compute exponents of qtilde^(-(n-p)/2) 
        detTensorGridΣθ = -0.5 * logdet((choleskyΣθ)) #compute exponents of |Σθ|^(-1/2) and |X'ΣθX|^(-1/2) 
        detTensorGridXΣX = -1.0 * sum(log.(diag(choleskyXΣX.U))) 
        jacTensorGrid = (1-p/n) * jacvals[r2]
        priorTensorGrid = logProb(btg.priorθ, t1) + logProb(btg.priorλ, t2)
        powerGrid[I] = qTensorGrid + detTensorGridΣθ + detTensorGridXΣX + jacTensorGrid + priorTensorGrid #sum of exponents
    end
    powerGrid = exp.(powerGrid .- maximum(powerGrid)) #linear scaling
    weightsTensorGrid = (endswith(btg.quadType[1], "MonteCarlo") && endswith(btg.quadType[2], "MonteCarlo")) ? powerGrid : weightsTensorGrid .* powerGrid 
    weightsTensorGrid = weightsTensorGrid/sum(weightsTensorGrid) #normalized grid of weights
    return weightsTensorGrid
end

"""
Compute pdf and cdf functions
"""
function prediction_comp(btg::btg, weightsTensorGrid::Array{Float64}; validate = 0) #depends on both train_data and test_data
    nt1 = getNumLengthScales(btg.nodesWeightsθ) # number of dimensions of theta
    nt2 = getNum(btg.nodesWeightsθ) #number of theta quadrature in each dimension
    nl2 = getNum(btg.nodesWeightsλ) #number of lambda quadrature in each dimension
    #preallocate some space to store dpdf, pdf, and cdf functions, as well as location parameters, for all (θ, λ) quadrature node combinations
    if btg.quadType == ["Gaussian", "Gaussian"]
        tgridpdfderiv = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid)) #total num of length scales is num length scales of theta +1, because lambda is 1D
        tgridpdf = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid))
        tgridcdf = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid))
        tgridm = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid))
        tgridsigma_m = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid))
        tgridquantile = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid)) # store quantile of each T component

    elseif endswith(btg.quadType[1], "MonteCarlo") && endswith(btg.quadType[2], "MonteCarlo")
        tgridpdfderiv = Array{Function, 1}(undef, Base.size(weightsTensorGrid)) #total num of length scales is num length scales of theta +1, because lambda is 1D
        tgridpdf = Array{Function, 1}(undef, Base.size(weightsTensorGrid))
        tgridcdf = Array{Function, 1}(undef, Base.size(weightsTensorGrid))
        tgridm = Array{Function, 1}(undef, Base.size(weightsTensorGrid))
        tgridsigma_m = Array{Function, 1}(undef, Base.size(weightsTensorGrid))
        tgridquantile = Array{Function, 1}(undef, Base.size(weightsTensorGrid))
    else
        tgridpdfderiv = Array{Function, 2}(undef, Base.size(weightsTensorGrid)) #total num of length scales is num length scales of theta +1, because lambda is 1D
        tgridpdf = Array{Function, 2}(undef, Base.size(weightsTensorGrid))
        tgridcdf = Array{Function, 2}(undef, Base.size(weightsTensorGrid))
        tgridm = Array{Function, 2}(undef, Base.size(weightsTensorGrid))
        tgridsigma_m = Array{Function, 2}(undef, Base.size(weightsTensorGrid))
        tgridquantile = Array{Function, 2}(undef, Base.size(weightsTensorGrid))
    end
    R = CartesianIndices(weightsTensorGrid)
    for I in R
        r1 = (endswith(btg.quadType[1], "MonteCarlo") && endswith(btg.quadType[2], "MonteCarlo")) ? I : Tuple(I)[1:end-1] 
        r2 = Tuple(I)[end]
        θ = btg.quadType[1] == "Gaussian" ? getNodeSequence(getNodes(btg.nodesWeightsθ), r1) : getNodes(btg.nodesWeightsθ)[:, r1[1]]
        λ = getNodes(btg.nodesWeightsλ)[:, r2] 
        (dpdf, pdf, cdf, _, m, sigma_m, quantile_fun) = comp_tdist(btg, θ, λ)
        tgridpdfderiv[I] = dpdf
        tgridpdf[I] = pdf
        tgridcdf[I] = cdf
        tgridm[I] = m
        tgridsigma_m[I] = sigma_m
        tgridquantile[I] = quantile_fun # function of (x0, Fx0, q)
    end
    # store 
    function checkInput(x0, Fx0, y0)
        @assert typeof(x0) <: Array{T, 2} where T <: Real
        @assert typeof(Fx0) <:Array{T, 2} where T <: Real
        @assert (typeof(y0) <:Array{T, 1} where T<: Real && length(y0) == size(x0, 1) == size(Fx0, 1)) || typeof(y0) <:Real
        return nothing
    end
   
    function generate_view(grid)
        if btg.quadType == ["Gaussian", "Gaussian"] # 3d grid
            view_grid = @view grid[[1:nt2 for i = 1:nt1]..., 1:nl2]
        elseif (endswith(btg.quadType[1], "MonteCarlo") && endswith(btg.quadType[2], "MonteCarlo")) # 1d grid
            view_grid = @view grid[:]
        else # 2d grid
            view_grid = @view grid[:,:]
        end
        return view_grid
    end
    grid_pdf_deriv = similar(weightsTensorGrid); view_pdf_deriv = generate_view(grid_pdf_deriv)
    grid_pdf = similar(weightsTensorGrid); view_pdf = generate_view(grid_pdf)
    grid_cdf = similar(weightsTensorGrid); view_cdf = generate_view(grid_cdf)
    grid_m = similar(weightsTensorGrid); view_m = generate_view(grid_m)
    grid_sigma_m = similar(weightsTensorGrid); view_sigma_m = generate_view(grid_sigma_m)
    grid_quantile = similar(weightsTensorGrid); view_quantile = generate_view(grid_quantile)

    function evalgrid!(f, view, x0, Fx0...)
        for I in R
            view[I] = f[I](x0, Fx0...) 
        end
        return nothing
    end

    #dpdf_evalgrid = (x0, Fx0, y0) -> evalgrid!(tgridpdfderiv, x0, Fx0, y0, @view grid_pdf_deriv[1:end for i in 1:ndims(weightsTensorGrid)])
    #pdf_evalgrid = (x0, Fx0, y0) -> evalgrid!(tgridpdf, x0, Fx0, y0, @view grid_pdf[1:end for i in 1:ndims(weightsTensorGrid)]) 
    #cdf_evalgrid = (x0, Fx0, y0) -> evalgrid!(tgridcdf, x0, Fx0, y0, @view grid_cdf[1:end for i in 1:ndims(weightsTensorGrid)])

    evalgrid_dpdf!(x0, Fx0, y0) = evalgrid!(tgridpdfderiv, view_pdf_deriv, x0, Fx0, y0)
    evalgrid_pdf!(x0, Fx0, y0) = evalgrid!(tgridpdf, view_pdf, x0, Fx0, y0)
    evalgrid_cdf!(x0, Fx0, y0) = evalgrid!(tgridcdf, view_cdf, x0, Fx0, y0)
    evalgrid_m!(x0, Fx0) = evalgrid!(tgridm, view_m, x0, Fx0)
    evalgrid_sigma_m!(x0, Fx0) = evalgrid!(tgridsigma_m, view_sigma_m, x0, Fx0)
    evalgrid_quantile!(x0, Fx0, q) = evalgrid!(tgridquantile, view_quantile, x0, Fx0, q)

    #below we write y0[1] instead of y0, because sometimes the output will have a box around it, due to matrix-operations in the internal implementation
    dpdf = (x0, Fx0, y0) -> (evalgrid_dpdf!(x0, Fx0, y0[1]); dot(grid_pdf_deriv, weightsTensorGrid))
    pdf = (x0, Fx0, y0) -> (evalgrid_pdf!(x0, Fx0, y0[1]); dot(grid_pdf, weightsTensorGrid))
    cdf = (x0, Fx0, y0) -> (evalgrid_cdf!(x0, Fx0, y0[1]); dot(grid_cdf, weightsTensorGrid))
    
    # compute estimated quantile
    EY = (x0, Fx0) -> (evalgrid_m!(x0, Fx0); dot(grid_m, weightsTensorGrid))
    EY2 = (x0, Fx0) -> (evalgrid_sigma_m!(x0, Fx0); dot(grid_sigma_m, weightsTensorGrid))
    quantile_range = (x0, Fx0, q) -> (evalgrid_quantile!(x0, Fx0, q); grid_quantile)

    VarY =  (x0, Fx0) -> EY2(x0, Fx0) - (EY(x0, Fx0))^2
    v = btg.trainingData.n - btg.trainingData.p
    sigma_Y = (x0, Fx0) -> sqrt(VarY(x0, Fx0) * (v-2)/v)
    quant_estimt = (x0, Fx0, q) -> sigma_Y(x0, Fx0) * tdistinvcdf(v, q) + EY(x0, Fx0)
    quantInfo = (quant_estimt, EY, EY2, sigma_Y, quantile_range)
    
    return (pdf, cdf, dpdf, quantInfo) 

end

