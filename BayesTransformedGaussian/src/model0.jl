# include("transforms/transforms.jl")
# include("kernels/kernel.jl")
# include("priors/priors.jl")
# include("computation/buffers0.jl")
# include("dataStructs.jl")


#keep these
include("grids.jl")
include("iterator.jl")
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
    λbuffer_dict::Dict{T where T<:Real, λbuffer} #λbuffer stores gλvals and logjacvals
    train_buffer_dict::Union{Dict{Array{T, 1}, train_buffer}, Dict{T, train_buffer}} where T<: Real     #buffer for each theta value
    test_buffer_dict::Union{Dict{Array{T, 1} where T<: Real, test_buffer}, Dict{T where T<: Real, test_buffer}}  #buffer for each theta value  
    θλbuffer_dict::Union{Dict{Tuple{Array{T, 1}, T} where T<:Real , θλbuffer}, Dict{Tuple{T, T} where T<:Real , θλbuffer}} #key should be theta-lambda pair
    validation_λ_buffer_dict::Dict{Real, validation_λ_buffer} 
    validation_train_buffer_dict::Union{Dict{Array{T, 1} where  T<: Real, validation_train_buffer}, Dict{T where  T<: Real, validation_train_buffer}} 
    validation_test_buffer_dict::Union{Dict{Array{T, 1} where T<: Real, validation_test_buffer}, Dict{Real, validation_test_buffer}}
    validation_θλ_buffer_dict::Union{Dict{Tuple{Array{T, 1}, T} where T<:Real, validation_θλ_buffer}, Dict{Tuple{T, T} where T<:Real, validation_θλ_buffer}} 
    capacity::Int64 
    debug_log::Any
    debug_log2::Any
    function btg(trainingData::AbstractTrainingData, rangeθ, rangeλ; corr = Gaussian(), priorθ = Uniform(rangeθ), priorλ = Uniform(rangeλ), quadtype = ["Gaussian", "Gaussian"], transform = BoxCox())
        @assert typeof(corr)<:AbstractCorrelation
        @assert typeof(priorθ)<:priorType
        @assert typeof(priorλ)<:priorType
        @assert typeof(quadtype)<:Array{String,1}
        @assert typeof(transform)<: NonlinearTransform
        @assert Base.size(rangeθ, 1) == getDimension(trainingData) || Base.size(rangeθ, 1)==1
        nodesWeightsθ = nodesWeights(rangeθ, quadtype[1]; num_pts = 12, num_MC = 400)
        nodesWeightsλ = nodesWeights(rangeλ, quadtype[2]; num_pts = 12, num_MC = 400)
        λbuffer_dict = init_λbuffer_dict(nodesWeightsλ, trainingData, transform) 
        train_buffer_dict  = init_train_buffer_dict(nodesWeightsθ, trainingData, corr, quadtype[1]) #gets initialized upon initialization of btg object
        test_buffer_dict =  init_test_buffer_dict(nodesWeightsθ, train_buffer_dict) #dict of empty test_buffers, empty for now because we need testingData info supplied by function call to compute cross-covariances
        θλbuffer_dict = init_θλbuffer_dict(nodesWeightsθ, nodesWeightsλ, trainingData, λbuffer_dict, train_buffer_dict, quadtype) #Stores qtilde, βhat, Σθ_inv_y
        validation_train_buffer_dict = init_validation_train_buffer_dict(nodesWeightsθ) #empty dict, input is used to determine dictionary type
        validation_θλ_buffer_dict = init_validation_θλ_buffer_dict(nodesWeightsθ) #empty dict
        validation_test_buffer_dict =  init_validation_test_buffer_dict(nodesWeightsθ, test_buffer_dict) #empty dict
        validation_λ_buffer_dict = init_validation_λ_buffer_dict() #empty dict
        cap = getCapacity(trainingData)
        return new(trainingData, testingData(), trainingData.n, transform, corr, quadtype, priorθ, priorλ, nodesWeightsθ, nodesWeightsλ, λbuffer_dict, train_buffer_dict, test_buffer_dict, θλbuffer_dict, 
                    validation_λ_buffer_dict, validation_train_buffer_dict, validation_test_buffer_dict, validation_θλ_buffer_dict, cap, [], [])
    end
end
"""
Initialize validation buffers all at once
"""
function init_validation_buffers(btg::btg, train_buffer_dict::Union{Dict{Array{T, 1}, train_buffer}, Dict{T, train_buffer}} where T<: Real, θλbuffer_dict::Union{Dict{Tuple{Array{T, 1}, T} where T<:Real , θλbuffer}, Dict{Tuple{T, T} where T<:Real , θλbuffer}},
    test_buffer_dict::Union{Dict{Array{T, 1} where T<: Real, test_buffer}, Dict{T where T<: Real, test_buffer}}, λbuffer_dict::Dict{T where T<:Real, λbuffer}, i::Int64)  
    #println("Initializing validation buffers...")  
    for key in keys(train_buffer_dict)
        cur_train_buf = train_buffer_dict[key]
        push!(btg.validation_train_buffer_dict, key => validation_train_buffer(key::Union{Array{T, 1}, T} where T<:Real, i, cur_train_buf, btg.trainingData))
    end
    for key in keys(θλbuffer_dict) #key is tuple of the form (t1, t2), where t1 is theta-nodes (Float64 if single else Array{Float64}) and t2 is lambda node (Float64)
        cur_train_buf = train_buffer_dict[key[1]]
        cur_λbuf = λbuffer_dict[key[2]]
        cur_θλbuf = θλbuffer_dict[key]
        cur_validation_train_buf = btg.validation_train_buffer_dict[key[1]]
        push!(btg.validation_θλ_buffer_dict, key => validation_θλ_buffer(key[1]::Union{Array{T, 1}, T} where T<:Float64, key[2]::Float64, i, cur_train_buf::train_buffer, 
                                                                        cur_λbuf::λbuffer, cur_θλbuf::θλbuffer, cur_validation_train_buf::validation_train_buffer, btg.trainingData))
    end
    #for key in keys(test_buffer_dict)
    #    cur_train_buf = train_buffer_dict[key]
    #    cur_test_buf = test_buffer_dict[key]
    #    push!(btg.validation_test_buffer_dict, key => validation_test_buffer(key::Union{Array{T, 1}, T} where T<:Float64, cur_train_buf::train_buffer, cur_test_buf::test_buffer))
    #end
    for key in keys(λbuffer_dict)
        cur_λbuf = λbuffer_dict[key]
        push!(btg.validation_λ_buffer_dict, key=> validation_λ_buffer(cur_λbuf, i::Int64))
    end
end
#workflow is:
#1) set_test_data
#2) solve, i.e. get pdf and cdf 
#3) update_system if needed
#x, Fx, y

#workflow for validation is
#1) set test data
#2) solve once normally
#3) solve as many more times as needed with validation flag set to i, where i is the deleted point

#"""
#Updates btg object with newly observed data points. Used in the context of BO.
#"""
#function update!(btg::btg, x0, Fx0, y0) #extend system step, invariant is 
#    update!(btg.trainingData, x0, Fx0, y0)    
 #   update_train_buffer!(btg.train_buffer, btg.train)
#end

function solve(btg::btg; validate = 0)
    if validate != 0
        #@info "num keys theta lambda buffer", length(keys(btg.θλbuffer_dict))
        #println("btg.n: ", btg.n)
        @assert validate > 0 && btg.n>=validate
        init_validation_buffers(btg, btg.train_buffer_dict, btg.θλbuffer_dict, btg.test_buffer_dict, btg.λbuffer_dict, validate)
    end
    weightTensorGrid = weight_comp(btg; validate = validate)
    (pdf, cdf, dpdf, quantInfo, augmented_cdf_deriv) = prediction_comp(btg, weightTensorGrid; validate = validate)
end

"""
Stably compute weights in the mixture of T-distributions, i.e. |Σθ|^(-1/2) * |X'ΣθX|^(-1/2) * qtilde^(-(n-p)/2) * Jac(z)^(1-p/n) * pθ(θ) * pλ(λ),
for all combinations of quadrature nodes in θ and λ
"""
function weight_comp(btg::btg; validate = 0)#depends on train_data and not test_data
    nwθ = btg.nodesWeightsθ; nwλ = btg.nodesWeightsλ; quadType = btg.quadType
    nt1 = nwθ.d   #number of dimensions of theta 
    nt2 = nwθ.num #number of theta quadrature in each dimension
    nl1 = nwλ.d   #number of dimensions of lambda 
    nl2 = nwλ.num #number of lambda quadrature in each dimension
    n =  btg.trainingData.n; p = btg.trainingData.p #number of training points and dimension of covariates
    n = validate == 0 ? n : n-1 #for cross-validation, one point is deleted
    #all the logic from create_btg_iterator used to be here
    R, weightsTensorGrid = get_btg_iterator(nwθ, nwλ, quadType)
    powerGrid = similar(weightsTensorGrid) #used to store exponents of qtilde, determinants, jacobians, etc.
    for I in R
        (r1, r2, t1, t2) = get_index_slices(nwθ,nwλ, quadType, I)
        t1 = length(t1)==1 ? t1[1] : t1
        θλpair = (t1, t2)::Tuple{Union{Array{T, 1}, T}, T} where T<:Real #key composed of t1 and t2
        if validate == 0 #bring appropriate quantities into local scope
            (_, _, βhat, qtilde) = unpack(btg.θλbuffer_dict[θλpair])
            (_, _, _, _, _, logdetΣθ, logdetXΣX) = unpack(btg.train_buffer_dict[t1])
            (_, _, logjacval) = unpack(btg.λbuffer_dict[t2]) 
        else #validation values
            (_, _, _, βhat, qtilde) = unpack(btg.validation_θλ_buffer_dict[θλpair]) 
            (_, _, _, logdetΣθ, logdetXΣX) = unpack(btg.validation_train_buffer_dict[t1]) 
            logjacval = unpack(btg.validation_λ_buffer_dict[t2])
        end 
        p = length(βhat)
        qTensorGrid_I = -(n-p)/2 * log(qtilde[1])  #compute exponents of qtilde^(-(n-p)/2) 
        detTensorGridΣθ_I = -0.5 * logdetΣθ #compute exponents of |Σθ|^(-1/2) and |X'ΣθX|^(-1/2) 
        detTensorGridXΣX_I = -0.5 * logdetXΣX
        jacTensorGrid_I = (1-p/n) * logjacval
        priorTensorGrid_I = logProb(btg.priorθ, t1) + logProb(btg.priorλ, t2)
        powerGrid[I] = qTensorGrid_I + detTensorGridΣθ_I + detTensorGridXΣX_I + jacTensorGrid_I + priorTensorGrid_I #sum of exponents        
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
    nwθ = btg.nodesWeightsθ
    nt1 = getNumLengthScales(nwθ) # number of dimensions of theta
    nt2 = getNum(nwθ) #number of theta quadrature in each dimension
    nl2 = getNum(btg.nodesWeightsλ) #number of lambda quadrature in each dimension
    quadType = btg.quadType
    (tgridpdfderiv, tgridpdf, tgridcdf, tgridm, tgridsigma_m, tgridquantile, tgridcdf_augmented_deriv) = tgrids(nt1, nt2, nl2, quadType, weightsTensorGrid)
    R = CartesianIndices(weightsTensorGrid)
    for I in R #it would be nice to iterate over all the lambdas for theta before going to the next theta
        (r1, r2, θ, λ) = get_index_slices(btg.nodesWeightsθ, btg.nodesWeightsλ, quadType, I)
        (dpdf, pdf, cdf, cdf_augmented_deriv, m, sigma_m, quantile_fun) = comp_tdist(btg, θ, λ; validate = validate)
        tgridpdfderiv[I] = dpdf
        tgridpdf[I] = pdf
        tgridcdf[I] = cdf
        tgridm[I] = m
        tgridsigma_m[I] = sigma_m
        tgridquantile[I] = quantile_fun # function of (x0, Fx0, q)
        tgridcdf_augmented_deriv[I] = cdf_augmented_deriv
    end
    grid_pdf_deriv = similar(weightsTensorGrid); view_pdf_deriv = generate_view(grid_pdf_deriv, nt1, nt2, nl2, quadType)
    grid_pdf = similar(weightsTensorGrid); view_pdf = generate_view(grid_pdf, nt1, nt2, nl2, quadType)
    grid_cdf = similar(weightsTensorGrid); view_cdf = generate_view(grid_cdf, nt1, nt2, nl2, quadType)
    grid_m = similar(weightsTensorGrid); view_m = generate_view(grid_m, nt1, nt2, nl2, quadType)
    grid_sigma_m = similar(weightsTensorGrid); view_sigma_m = generate_view(grid_sigma_m, nt1, nt2, nl2, quadType)
    grid_quantile = similar(weightsTensorGrid); view_quantile = generate_view(grid_quantile, nt1, nt2, nl2, quadType)
    grid_augmented_deriv = similar(weightsTensorGrid); view_augmented_deriv = generate_view(grid_augmented_deriv, nt1, nt2, nl2, quadType)
    function evalgrid!(f, view, x0, Fx0...)
        for I in R
            view[I] = f[I](x0, Fx0...)[1] 
        end
        return nothing
    end
    evalgrid_dpdf!(x0, Fx0, y0) = evalgrid!(tgridpdfderiv, view_pdf_deriv, x0, Fx0, y0)
    evalgrid_pdf!(x0, Fx0, y0) = evalgrid!(tgridpdf, view_pdf, x0, Fx0, y0)
    evalgrid_cdf!(x0, Fx0, y0) = evalgrid!(tgridcdf, view_cdf, x0, Fx0, y0)
    evalgrid_m!(x0, Fx0) = evalgrid!(tgridm, view_m, x0, Fx0)
    evalgrid_sigma_m!(x0, Fx0) = evalgrid!(tgridsigma_m, view_sigma_m, x0, Fx0)
    evalgrid_quantile!(x0, Fx0, q) = evalgrid!(tgridquantile, view_quantile, x0, Fx0, q)
    evalgrid_augmented_deriv!(x0, Fx0, y0) = evalgrid!(tgridcdf_augmented_deriv, view_augmented_deriv, x0, Fx0, y0)

    function checkInput(x0, Fx0, y0)
        try 
        @assert typeof(x0) <: Array{T, 2} where T <: Real
        @assert typeof(Fx0) <:Array{T, 2} where T <: Real
        catch e
            @info "x0", x0
            @info "Fx0", Fx0
        end
        #@assert typeof(y0) <:Array{T, 1} where T<: Real 
        #@assert(length(y0) == size(x0, 1))
        #&&  == size(Fx0, 1)) || typeof(y0) <:Real
        return nothing
    end
    #below we write y0[1] instead of y0, because sometimes the output will have a box around it, due to matrix-operations in the internal implementation
    dpdf = (x0, Fx0, y0) -> (btg.debug_log = []; checkInput(x0, Fx0, y0); evalgrid_dpdf!(x0, Fx0, y0[1]); dot(grid_pdf_deriv, weightsTensorGrid))
    pdf = (x0, Fx0, y0) -> (btg.debug_log = []; checkInput(x0, Fx0, y0); evalgrid_pdf!(x0, Fx0, y0[1]); dot(grid_pdf, weightsTensorGrid))
    cdf = (x0, Fx0, y0) -> (btg.debug_log = []; checkInput(x0, Fx0, y0); evalgrid_cdf!(x0, Fx0, y0[1]); dot(grid_cdf, weightsTensorGrid))
    augmented_cdf_deriv = (x0, Fx0, y0) -> (btg.debug_log = []; checkInput(x0, Fx0, y0); evalgrid_augmented_deriv!(x0, Fx0, y0[1]); dot(grid_augmented_deriv, weightsTensorGrid))

    # compute estimated quantile
    EY = (x0, Fx0) -> (evalgrid_m!(x0, Fx0); dot(grid_m, weightsTensorGrid))
    EY2 = (x0, Fx0) -> (evalgrid_sigma_m!(x0, Fx0); dot(grid_sigma_m, weightsTensorGrid))
    quantile_range = (x0, Fx0, q) -> (evalgrid_quantile!(x0, Fx0, q); grid_quantile)

    VarY =  (x0, Fx0) -> EY2(x0, Fx0) - (EY(x0, Fx0))^2
    v = btg.trainingData.n - btg.trainingData.p
    sigma_Y = (x0, Fx0) -> sqrt(VarY(x0, Fx0) * (v-2)/v)
    quant_estimt = (x0, Fx0, q) -> sigma_Y(x0, Fx0) * tdistinvcdf(v, q) + EY(x0, Fx0)
    quantInfo = (quant_estimt, EY, EY2, sigma_Y, quantile_range)
    
    return (pdf, cdf, dpdf, quantInfo, augmented_cdf_deriv) 
end

