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
    quadType::String #Gaussian, Turan, or MonteCarlo
    priorθ::priorType
    priorλ::priorType
    nodesWeightsθ #integration nodes and weights for θ
    nodesWeightsλ #integration nodes and weights for λ; nodes and weights should remain constant throughout the lifetime of the btg object
    train_buffer_dict::Dict{Union{Array{T, 1}, T} where T<: Real, train_buffer}   #buffer for each theta value
    test_buffer_dict::Dict{Union{Array{T, 1}, T} where T<: Real, test_buffer}  #buffer for each theta value
    capacity::Int64
    function btg(trainingData::AbstractTrainingData, rangeθ, rangeλ; corr = Gaussian(), priorθ = Uniform(rangeθ), priorλ = Uniform(rangeλ), quadtype = "Gaussian", transform = BoxCox())
        @assert typeof(corr)<:AbstractCorrelation
        @assert typeof(priorθ)<:priorType
        @assert typeof(priorλ)<:priorType
        @assert typeof(quadtype)<:String
        @assert typeof(transform)<: NonlinearTransform
        @assert Base.size(rangeθ, 1) == getDimension(trainingData) || Base.size(rangeθ, 1)==1
        #a btg object really should contain a bunch of train buffers correpsonding to different theta-values
        #we should add some fields to the nodesweights_theta data structure to figure out the number of dimensions we are integrating over...should we allow different length scale ranges w/ different quadrature nodes? I think so??
        nodesWeightsθ = nodesWeights(rangeθ, quadtype)
        nodesWeightsλ = nodesWeights(rangeλ, quadtype)
        train_buffer_dict  = init_train_buffer_dict(nodesWeightsθ, trainingData, corr)
        test_buffer_dict = Dict{Union{Array{T, 1}, T} where T<:Real, test_buffer}(arr => test_buffer() for arr in keys(train_buffer_dict)) #initialize keys of dict with unitialized test buffer values
        cap = getCapacity(trainingData)
        new(trainingData, testingData(), 0, transform, corr, quadtype, priorθ, priorλ, nodesWeightsθ, nodesWeightsλ, train_buffer_dict, test_buffer_dict, cap)
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

function solve(btg::btg)
    weightTensorGrid = weight_comp(btg)
<<<<<<< HEAD
    (dpdf, pdf, cdf, TmixInfo) = prediction_comp(btg, weightTensorGrid)
=======
    (pdf, cdf, dpdf) = prediction_comp(btg, weightTensorGrid)
>>>>>>> ae145666b5d3486499db2356e2f7dbd1e6508ba3
end

"""
Stably compute weights in the mixture of T-distributions, i.e. |Σθ|^(-1/2) * |X'ΣθX|^(-1/2) * qtilde^(-(n-p)/2) * Jac(z)^(1-p/n) * pθ(θ) * pλ(λ),
for all combinations of quadrature nodes in θ and λ
"""
function weight_comp(btg::btg)#depends on train_data and not test_data
    # line 36-99 in tensorgrid.jl 
    nd = btg.nodesWeightsθ.d + 1 #number of dimensions of theta (d) plus number dimensions of lambda (1)
    nq = btg.nodesWeightsθ.num #number of quadrature nodes
    n =  btg.trainingData.n; p = btg.trainingData.p #number of training points and dimension of covariates
    weightsTensorGrid = Array{Float64, nd}(undef, Tuple([nq for i = 1:nd])) #initialize tensor grid
    R = CartesianIndices(weightsTensorGrid)
    for I in R #I is multi-index
        weightsTensorGrid[I] = getProd(vcat(btg.nodesWeightsθ.weights, btg.nodesWeightsλ.weights), I) #this step can be simplified because the tensor is symmetric (weights are the same along each dimension)
    end

    #compute exponents of qtilde^(-(n-p)/2)
    qTensorGrid = similar(weightsTensorGrid)
    z = getLabel(btg.trainingData)
    g = (x, λ) -> btg.g(x, λ); dg = (x, λ) -> partialx(btg.g, x, λ)
    lmbda = λ -> g(z, λ)
    gλvals = Array{Float64, 2}(undef, length(z), nq) #preallocate space to store gλz arrays
    for i = 1:nq 
        gλvals[:, i] = lmbda(btg.nodesWeightsλ.nodes[i])
    end
    Fx = getCovariates(btg.trainingData)
    for I in R
        train_buffer = btg.train_buffer_dict[getNodeSequence(getNodes(btg.nodesWeightsθ), Tuple(I)[1:end-1])] #look up train buffer based on combination of theta quadrature nodes
        choleskyXΣX = train_buffer.choleskyXΣX
        choleskyΣθ = train_buffer.choleskyΣθ
        gλz = gλvals[:, Tuple(I)[end]]
        βhat = choleskyXΣX\(Fx'*(choleskyΣθ\gλz)) 
        qtilde = (expr = gλz-Fx*βhat; expr'*(choleskyΣθ\expr))
        qTensorGrid[I] = -(n-p)/2 * log(qtilde)
    end

    #compute exponents of |Σθ|^(-1/2) and |X'ΣθX|^(-1/2)  
    detTensorGridΣθ = similar(weightsTensorGrid)
    detTensorGridXΣX = similar(weightsTensorGrid) 
    for I in R
        train_buffer = btg.train_buffer_dict[getNodeSequence(getNodes(btg.nodesWeightsθ), Tuple(I)[1:end-1])] #look up train buffer based on combination of theta quadrature nodes
        choleskyΣθ = train_buffer.choleskyΣθ
        choleskyXΣX = train_buffer.choleskyXΣX
        detTensorGridΣθ[I] = -0.5 * logdet((choleskyΣθ)) #log determinant of incremental cholesky
        detTensorGridXΣX[I] = -1.0 * sum(log.(diag(choleskyXΣX.U))) 
    end
    
    #compute exponents of Jac(z)^(1-p/n)
    jacTensorGrid = similar(weightsTensorGrid)
    jacvals = zeros(1, nq)
    for i = 1:nq
        jacvals[i] = sum(log.(abs.(map( x-> dg(x, btg.nodesWeightsλ.nodes[i]), z))))
    end
    for I in R
        jacTensorGrid[I] = (1-p/n) * jacvals[Tuple(I)[end]]
    end
    
    #compute exponents of pθ(θ)*pλ(λ)
    priorTensorGrid = similar(weightsTensorGrid)
    for I in R
        tup = Tuple(I); r1 = tup[1:end-1]; r2 = tup[end]
        t1 = getNodeSequence(getNodes(btg.nodesWeightsθ), r1)
        t2 = getNodeSequence(getNodes(btg.nodesWeightsλ), r2)
        priorTensorGrid[I] = logProb(btg.priorθ, t1) + logProb(btg.priorλ, t2)
    end
    @assert Base.size(qTensorGrid) == Base.size(detTensorGridΣθ) == Base.size(detTensorGridXΣX) == Base.size(jacTensorGrid) == Base.size(priorTensorGrid)
    powerGrid =  qTensorGrid + detTensorGridΣθ + detTensorGridXΣX + jacTensorGrid + priorTensorGrid #sum of exponents
    powerGrid = exp.(powerGrid .- maximum(powerGrid)) #linear scaling
    weightsTensorGrid = powerGrid .* weightsTensorGrid
    weightsTensorGrid =  weightsTensorGrid/sum(weightsTensorGrid) #normalized grid of weights
    return weightsTensorGrid
end

"""
Compute pdf and cdf functions
"""
function prediction_comp(btg::btg, weightsTensorGrid::Array{Float64}) #depends on both train_data and test_data
    #preallocate some space to store dpdf, pdf, and cdf functions, as well as location parameters, for all (θ, λ) quadrature node combinations
    tgridpdfderiv = Array{Function, getNumLengthScales(btg.nodesWeightsθ) + 1}(undef, Base.size(weightsTensorGrid)) #total num of length scales is num length scales of theta +1, because lambda is 1D
    tgridpdf = Array{Function, getNumLengthScales(btg.nodesWeightsθ) +1}(undef, Base.size(weightsTensorGrid))
    tgridcdf = Array{Function, getNumLengthScales(btg.nodesWeightsθ)+1 }(undef, Base.size(weightsTensorGrid))
    # tgridm = Array{Function, getNumLengthScales(btg.nodesWeightsθ)+1 }(undef, Base.size(weightsTensorGrid))
    # tgridsigma_m = Array{Function, getNumLengthScales(btg.nodesWeightsθ)+1}(undef, Base.size(weightsTensorGrid))

    #similar(weightsTensorGrid) 
    #tgridpdf = similar(weightsTensorGrid) 
    #tgridcdf = similar(weightsTensorGrid)

    R = CartesianIndices(weightsTensorGrid)
    for I in R
        θ = getNodeSequence(getNodes(btg.nodesWeightsθ), Tuple(I)[1:end-1]) 
        λ = getNodeSequence(getNodes(btg.nodesWeightsλ), Tuple(I)[end]) 
        (dpdf, pdf, cdf) = comp_tdist(btg, θ, λ)
        tgridpdfderiv[I] = dpdf
        tgridpdf[I] = pdf
        tgridcdf[I] = cdf
        # tgridm[I] = m
        # tgridsigma_m[I] = sigma_m
    end
    # store 
    function checkInput(x0, Fx0, y0)
        @assert typeof(x0) <: Array{T, 2} where T <: Real
        @assert typeof(Fx0) <:Array{T, 2} where T <: Real
        @assert (typeof(y0) <:Array{T, 1} where T<: Real && length(y0) == size(x0, 1) == size(Fx0, 1)) || typeof(y0) <:Real
        return nothing
    end

    d = ndims(weightsTensorGrid) #replace end
    nq = getNum(btg.nodesWeightsθ)
    grid_pdf_deriv = similar(weightsTensorGrid); view_pdf_deriv = @view grid_pdf_deriv[[1:nq for i = 1:d]...]
    grid_pdf = similar(weightsTensorGrid); view_pdf = @view grid_pdf[[1:nq for i = 1:d]...]
    grid_cdf = similar(weightsTensorGrid); view_cdf =  @view grid_cdf[[1:nq for i = 1:d]...]
    # grid_m = similar(weightsTensorGrid); view_m =  @view grid_m[[1:nq for i = 1:d]...]
    # grid_sigma_m = similar(weightsTensorGrid); view_sigma_m =  @view grid_m[[1:nq for i = 1:d]...]
    
    function evalgrid!(f, x0, Fx0, y0, view)
        checkInput(x0, Fx0, y0) 
        for I in R
            view[I] = f[I](x0, Fx0, y0) 
        end
        return nothing
    end

    function evalgrid_quant!(f, x0, Fx0, view)
        # checkInput(x0, Fx0, y0) 
        for I in R
            view[I] = f[I](x0, Fx0) 
        end
        return nothing
    end

    #dpdf_evalgrid = (x0, Fx0, y0) -> evalgrid!(tgridpdfderiv, x0, Fx0, y0, @view grid_pdf_deriv[1:end for i in 1:ndims(weightsTensorGrid)])
    #pdf_evalgrid = (x0, Fx0, y0) -> evalgrid!(tgridpdf, x0, Fx0, y0, @view grid_pdf[1:end for i in 1:ndims(weightsTensorGrid)]) 
    #cdf_evalgrid = (x0, Fx0, y0) -> evalgrid!(tgridcdf, x0, Fx0, y0, @view grid_cdf[1:end for i in 1:ndims(weightsTensorGrid)])

    evalgrid_dpdf!(x0, Fx0, y0) = evalgrid!(tgridpdfderiv, x0, Fx0, y0, view_pdf_deriv)
    evalgrid_pdf!(x0, Fx0, y0) = evalgrid!(tgridpdf, x0, Fx0, y0, view_pdf)
    evalgrid_cdf!(x0, Fx0, y0) = evalgrid!(tgridcdf, x0, Fx0, y0, view_cdf)
    # evalgrid_m!(x0, Fx0) = evalgrid_quant!(tgridm, x0, Fx0, view_m)
    # evalgrid_sigma_m!(x0, Fx0) = evalgrid_quant!(tgridsigma_m, x0, Fx0, view_sigma_m)

    dpdf = (x0, Fx0, y0) -> (evalgrid_dpdf!(x0, Fx0, y0); dot(grid_pdf_deriv, weightsTensorGrid))
    pdf = (x0, Fx0, y0) -> (evalgrid_pdf!(x0, Fx0, y0); dot(grid_pdf, weightsTensorGrid))
    cdf = (x0, Fx0, y0) -> (evalgrid_cdf!(x0, Fx0, y0); dot(grid_cdf, weightsTensorGrid))
    return (dpdf, pdf, cdf)
    # compute estimated quantile
    # x_ref = (x0, Fx0) -> (evalgrid_m!(x0, Fx0); dot(grid_m, weightsTensorGrid))
    # var_ref1 = (x0, Fx0) -> (evalgrid_sigma_m!(x0, Fx0); dot(grid_sigma_m, weightsTensorGrid))
    # var_ref =  (x0, Fx0) -> var_ref1(x0, Fx0) - (x_ref(x0, Fx0))^2
    # sigma_mix = (x0, Fx0) -> sqrt(var_ref(x0, Fx0) * (v-2)/v)
    # v = btg.trainingData.n - btg.trainingData.p
    # # sigma_mix = (x0, Fx0) -> sqrt(var_ref(x0, Fx0) * (v-2)/v)
    # # TmixInfo = [x_ref, sigma_mix, v]
    # quant_estimt = (x0, Fx0, q) -> sigma_mix(x0, Fx0) * tdistinvcdf(v, q) + x_ref(x0, Fx0)
    # return (dpdf, pdf, cdf, quant_estimt, sigma_mix, x_ref) 

end

