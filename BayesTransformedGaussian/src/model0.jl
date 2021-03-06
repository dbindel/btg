# include("transforms/transforms.jl")
# include("kernels/kernel.jl")
# include("priors/priors.jl")
# include("computation/buffers0.jl")
# include("dataStructs.jl")


#keep these
include("grids.jl")
include("iterator.jl")
using StatsFuns
using TimerOutputs
if !@isdefined(to)
    const to = TimerOutput()
end
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
    # g::NonlinearTransform #transform family, e.g. BoxCox()
    # k::AbstractCorrelation  #kernel family, e.g. Gaussian()
    # quadType::Array{String,1} #type for theta and lambda respectively, Gaussian, Turan, or MonteCarlo
    # priorθ::priorType
    # priorλ::priorType
    options::BtgOptions
    nodesWeightsθ #integration nodes and weights for θ
    nodesWeightsλ #integration nodes and weights for λ; nodes and weights should remain constant throughout the lifetime of the btg object
    λbuffer_dict::Dict{T where T<:Real, λbuffer} #λbuffer stores gλvals and logjacvals
    train_buffer_dict::Union{Dict{Array{T, 1}, train_buffer}, Dict{T, train_buffer}} where T<: Real     #buffer for each theta value
    test_buffer_dict::Union{Dict{Array{T, 1} where T<: Real, test_buffer}, Dict{T where T<: Real, test_buffer}}  #buffer for each theta value  
    jac_buffer_dict::Union{Dict{Array{T, 1} where T<: Real, jac_buffer}, Dict{T where T<: Real, jac_buffer}} #buffer for storing Jacobians
    θλbuffer_dict::Union{Dict{Tuple{Array{T, 1}, T} where T<:Real , θλbuffer}, Dict{Tuple{T, T} where T<:Real , θλbuffer}} #key should be theta-lambda pair
    validation_λ_buffer_dict::Dict{Real, validation_λ_buffer} 
    validation_train_buffer_dict::Union{Dict{Array{T, 1} where  T<: Real, validation_train_buffer}, Dict{T where  T<: Real, validation_train_buffer}} 
    validation_test_buffer_dict::Union{Dict{Array{T, 1} where T<: Real, validation_test_buffer}, Dict{Real, validation_test_buffer}}
    validation_θλ_buffer_dict::Union{Dict{Tuple{Array{T, 1}, T} where T<:Real, validation_θλ_buffer}, Dict{Tuple{T, T} where T<:Real, validation_θλ_buffer}} 
    capacity::Int64 
    debug_log::Any
    debug_log2::Any
    #weightsTensorGrid::Union{Array{T}, G} where T<:Float64 where G<:Nothing
    function btg(trainingData::AbstractTrainingData, options::BtgOptions)
        rangeθ = options.parameter_range["θ"] 
        rangeλ = options.parameter_range["λ"]
        corr = getfield(Main, Symbol(options.kernel_type))()
        priorθ = options.parameter_prior["θ"]
        priorλ = options.parameter_prior["λ"]
        quadtype = [options.quadrature_type["θ"], options.quadrature_type["λ"]]
        transform = getfield(Main, Symbol(options.transform_type))()
        num_gq = options.quadrature_size["Gaussian"]
        num_mc = options.quadrature_size["MonteCarlo"]
        @timeit to "assert statements, type-checking" begin
            @assert typeof(corr)<:AbstractCorrelation
            @assert typeof(priorθ)<:priorType
            @assert typeof(priorλ)<:priorType
            @assert typeof(quadtype)<:Array{String,1}
            @assert typeof(transform)<: NonlinearTransform
        end
        @timeit to "nodesWeightsθ" nodesWeightsθ = nodesWeights("θ", rangeθ, rangeλ, quadtype[1]; num_pts = num_gq, num_MC = num_mc)
        @timeit to "nodesWeightsλ" nodesWeightsλ = nodesWeights("λ", rangeλ, rangeθ, quadtype[2]; num_pts = num_gq, num_MC = num_mc)
        @timeit to "init λbuffer_dict" λbuffer_dict = init_λbuffer_dict(nodesWeightsλ, trainingData, transform) 
        @timeit to "init train buffer dict" train_buffer_dict = init_train_buffer_dict(nodesWeightsθ, trainingData, corr, quadtype[1]) #gets initialized upon initialization of btg object
        @timeit to "test_buffer_dict" test_buffer_dict =  init_empty_buffer_dict(nodesWeightsθ, train_buffer_dict, test_buffer) #dict of empty test_buffers, empty for now because we need testingData info supplied by function call to compute cross-covariances
        @timeit to "jac_buffer_dict" jac_buffer_dict = init_empty_buffer_dict(nodesWeightsθ, train_buffer_dict, jac_buffer)
        @timeit to "θλbuffer_dict" θλbuffer_dict = init_θλbuffer_dict(nodesWeightsθ, nodesWeightsλ, trainingData, λbuffer_dict, train_buffer_dict, quadtype) #Stores qtilde, βhat, Σθ_inv_y
        @timeit to "validation_train_buffer_dict" validation_train_buffer_dict = init_validation_train_buffer_dict(nodesWeightsθ) #empty dict, input is used to determine dictionary type
        @timeit to "validation_θλ_buffer_dict" validation_θλ_buffer_dict = init_validation_θλ_buffer_dict(nodesWeightsθ) #empty dict
        @timeit to "validation_test_buffer_dict" validation_test_buffer_dict =  init_validation_test_buffer_dict(nodesWeightsθ, test_buffer_dict) #empty dict
        @timeit to "validation_λ_buffer_dict" validation_λ_buffer_dict = init_validation_λ_buffer_dict() #empty dict
        cap = getCapacity(trainingData)
        new(trainingData, testingData(), trainingData.n, options, nodesWeightsθ, nodesWeightsλ, λbuffer_dict, train_buffer_dict, test_buffer_dict,  
        jac_buffer_dict, θλbuffer_dict, validation_λ_buffer_dict, validation_train_buffer_dict, validation_test_buffer_dict, validation_θλ_buffer_dict, cap, [], [])
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

"""
BtgModel object contains results from solve
"""
mutable struct btgModel
    pdf::Function
    cdf::Function
    dpdf::Function
    augmented_cdf_deriv::Function
    augmented_cdf_hess::Function
    quantInfo::Function
    tgridpdf::Array{Function,2}
    tgridcdf::Array{Function,2}
    tgridm::Array{Function,2}
    tgridsigma_m::Array{Function,2}
    weightsTensorGrid::Array{T,2} where T<:Real
    function btgModel(btg::btg; validate = 0, derivatives = false)
        #println("validate in solve_btg is", validate)
        @assert typeof(derivatives) == Bool
        @assert typeof(validate) == Int64
        #@info "derivatives", derivatives
        if validate != 0 && derivatives == true
            println("Derivatives not supported in cross-validation (when validate flag > 0).")
            return
        end
        if validate != 0
            #@info "num keys theta lambda buffer", length(keys(btg.θλbuffer_dict))
            #println("btg.n: ", btg.n)
            @assert validate > 0 && btg.n>=validate
            init_validation_buffers(btg, btg.train_buffer_dict, btg.θλbuffer_dict, btg.test_buffer_dict, btg.λbuffer_dict, validate)
        end
        @timeit to "compute weights" weightsTensorGrid = weight_comp(btg; validate = validate)
        println("\n weightsTensorGrid")
        display(weightsTensorGrid)
        #(pdf, cdf, dpdf, augmented_cdf_deriv, augmented_cdf_hess, quantInfo) = prediction_comp(btg, weightsTensorGrid; validate = validate, derivatives = derivatives)
        (pdf, cdf, dpdf, augmented_cdf_deriv, augmented_cdf_hess, quantInfo, tgridpdf, tgridcdf, tgridm, tgridsigma_m) = prediction_comp(btg, weightsTensorGrid; validate = validate, derivatives = derivatives)
        # return (pdf, cdf, dpdf, augmented_cdf_deriv, augmented_cdf_hess, quantInfo, tgridpdf, tgridcdf, tgridm, tgridsigma_m, weightsTensorGrid) 
        new(pdf, cdf, dpdf, augmented_cdf_deriv, augmented_cdf_hess, quantInfo, tgridpdf, tgridcdf, tgridm, tgridsigma_m, weightsTensorGrid)
    end
end

"""
Stably compute weights in the mixture of T-distributions, i.e. |Σθ|^(-1/2) * |X'ΣθX|^(-1/2) * qtilde^(-(n-p)/2) * Jac(z)^(1-p/n) * pθ(θ) * pλ(λ),
for all combinations of quadrature nodes in θ and λ
"""
function weight_comp(btg::btg; validate = 0, debug = false)#depends on train_data and not test_data
    nwθ = btg.nodesWeightsθ; nwλ = btg.nodesWeightsλ
    quadType = [btg.options.quadrature_type["θ"], btg.options.quadrature_type["λ"]]
    nt1 = nwθ.d   #number of dimensions of theta 
    nt2 = nwθ.num #number of theta quadrature in each dimension
    nl1 = nwλ.d   #number of dimensions of lambda 
    nl2 = nwλ.num #number of lambda quadrature in each dimension
    n =  btg.trainingData.n; p = btg.trainingData.p #number of training points and dimension of covariates
    n = validate == 0 ? n : n-1 #for cross-validation, one point is deleted
    #all the logic from create_btg_iterator used to be here
    R, weightsTensorGrid = get_btg_iterator(nwθ, nwλ, quadType)
    powerGrid = similar(weightsTensorGrid) #used to store exponents of qtilde, determinants, jacobians, etc.
    qTensorGrid = similar(weightsTensorGrid)
    detTensorGridΣθ = similar(weightsTensorGrid)
    detTensorGridXΣX = similar(weightsTensorGrid)
    jacTensorGrid = similar(weightsTensorGrid)
    priorTensorGrid = similar(weightsTensorGrid)
    for I in R
        (r1, r2, t1, t2) = get_index_slices(nwθ, nwλ, quadType, I)
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
        priorTensorGrid_I = logProb(btg.options.parameter_prior["θ"], t1) + logProb(btg.options.parameter_prior["λ"], t2)
        powerGrid[I] = qTensorGrid_I + detTensorGridΣθ_I + detTensorGridXΣX_I + jacTensorGrid_I + priorTensorGrid_I #sum of exponents              
        #for testing purposes
        qTensorGrid[I] = qTensorGrid_I
        detTensorGridΣθ[I] = detTensorGridΣθ_I
        detTensorGridXΣX[I] = detTensorGridXΣX_I
        jacTensorGrid[I] = jacTensorGrid_I
        priorTensorGrid[I] = priorTensorGrid_I

    end
    temp_power = copy(powerGrid)
    powerGrid = exp.(powerGrid .- maximum(powerGrid)) #linear scaling
    weightsTensorGrid = (endswith(quadType[1], "MonteCarlo") && endswith(quadType[2], "MonteCarlo")) ? powerGrid : weightsTensorGrid .* powerGrid 
    weightsTensorGrid = weightsTensorGrid/sum(weightsTensorGrid) #normalized grid of weights
    #btg.weightsTensorGrid = weightsTensorGrid #store a copy of weights in BTG as well for debugging purposes
    
    # println("Log likelihood")
    # display(temp_power')
    # #print grids to detect over-concentration
    # println("qTensorGrid")
    # display(qTensorGrid)
    # println("detTensorGridΣθ")
    # display(detTensorGridΣθ)
    # println("detTensorGridXΣX")
    # display(detTensorGridXΣX)
    # println("jacTensorGrid")
    # display(jacTensorGrid)
    # println("weightsTensorGrid")
    # display(weightsTensorGrid)
    # println("priorTensorGrid")
    # display(priorTensorGrid)

    return weightsTensorGrid
end


function reset_test_buf(btg::btg)
    refresh_buffer_dict(btg.test_buffer_dict)
    return nothing
end

function reset_jac_buf(btg::btg)
    refresh_buffer_dict(btg.jac_buffer_dict)
    return nothing
end

"""
Compute pdf and cdf functions
"""
function prediction_comp(btg::btg, weightsTensorGrid::Array{Float64}; validate = 0, derivatives = false, debug = false) #depends on both train_data and test_data
    #@info "sum(weightsTensorGrid)", sum(weightsTensorGrid)
    nwθ = btg.nodesWeightsθ
    nt1 = getNumLengthScales(nwθ) # number of dimensions of theta
    nt2 = getNum(nwθ) #number of theta quadrature in each dimension
    nl2 = getNum(btg.nodesWeightsλ) #number of lambda quadrature in each dimension
    quadType = [btg.options.quadrature_type["θ"], btg.options.quadrature_type["λ"]]
    (tgridpdfderiv, tgridpdf, tgridcdf, tgridm, tgridsigma_m, tgridquantile, tgridcdf_augmented_deriv, tgridcdf_augmented_hess) = tgrids(nt1, nt2, nl2, quadType, weightsTensorGrid; derivatives = derivatives)
    R = CartesianIndices(weightsTensorGrid)
    @timeit to "put functions in array" begin
        for I in R #it would be nice to iterate over all the lambdas for theta before going to the next theta
            (r1, r2, θ, λ) = get_index_slices(btg.nodesWeightsθ, btg.nodesWeightsλ, quadType, I)
            #@info "validate in prediction_comp", validate
            (dpdf, pdf, cdf, cdf_jac_us, cdf_hess_us, q_fun, computeqmC) = comp_tdist(btg, θ, λ; validate = validate)
            tgridpdfderiv[I] = dpdf
            tgridpdf[I] = pdf
            tgridcdf[I] = cdf 
            m = (x0, Fx0, y) -> computeqmC(x0, Fx0)[1]
            sigma = (x0, Fx0, y) -> ( (_, _, _, s) = computeqmC(x0, Fx0); s)
            tgridm[I] = m
            tgridsigma_m[I] = sigma
            tgridquantile[I] = q_fun # function of (x0, Fx0, q)
            if derivatives
                tgridcdf_augmented_deriv[I] = cdf_jac_us
                tgridcdf_augmented_hess[I] = cdf_hess_us
            end
        end
    end

    grid_pdf_deriv = similar(weightsTensorGrid); view_pdf_deriv = generate_view(grid_pdf_deriv, nt1, nt2, nl2, quadType)
    grid_pdf = similar(weightsTensorGrid); view_pdf = generate_view(grid_pdf, nt1, nt2, nl2, quadType)
    grid_cdf = similar(weightsTensorGrid); view_cdf = generate_view(grid_cdf, nt1, nt2, nl2, quadType)
    grid_m = similar(weightsTensorGrid); view_m = generate_view(grid_m, nt1, nt2, nl2, quadType)
    grid_sigma_m = similar(weightsTensorGrid); view_sigma_m = generate_view(grid_sigma_m, nt1, nt2, nl2, quadType)
    grid_quantile = similar(weightsTensorGrid); view_quantile = generate_view(grid_quantile, nt1, nt2, nl2, quadType)

    grid_augmented_deriv = derivatives == true ? Array{Array{Real, 1}, ndims(weightsTensorGrid)}(undef, size(weightsTensorGrid)) : nothing #stores jacobians
    view_augmented_deriv = derivatives == true ? generate_view(grid_augmented_deriv, nt1, nt2, nl2, quadType) : nothing

    grid_augmented_hess = derivatives == true ? Array{Array{Real, 2}, ndims(weightsTensorGrid)}(undef, size(weightsTensorGrid)) : nothing #stores hessians
    view_augmented_hess = derivatives == true ? generate_view(grid_augmented_hess, nt1, nt2, nl2, quadType) : nothing

    function evalgrid!(f, view, x0, Fx0, y0)
        for I in R
            view[I] = (res = f[I](x0, Fx0, y0); 
            #@info "size of res", size(res);
            length(res)==1 ? res[1] : ( size(res, 1) == size(res, 2) && size(res, 1) > 1 ?  res : res[:] )) #unboxes scalar, flattens vector, keeps 2D array the same
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
    evalgrid_augmented_hess!(x0, Fx0, y0) = evalgrid!(tgridcdf_augmented_hess, view_augmented_hess, x0, Fx0, y0)

    function checkInput(x0, Fx0, y0)
        try 
        @assert typeof(x0) <: Array{T, 2} where T <: Real
        @assert typeof(Fx0) <:Array{T, 2} where T <: Real
        catch e
            #@info "x0", x0
            #@info "Fx0", Fx0
        end
        #@assert typeof(y0) <:Array{T, 1} where T<: Real 
        #@assert(length(y0) == size(x0, 1))
        #&&  == size(Fx0, 1)) || typeof(y0) <:Real
        return nothing
    end
    """
    Computes dot product of multi-dimensional array of arrays and multi-dimensional array of scalars
    Return type is array
    """
    function dot2(x:: Array{W} where W<:Array{G} where G<:Real, y::Array{P} where P<:Real)
        R = CartesianIndices(x)
        res = zeros(size(x[R[1]]))
        for I in R
            res = res + x[I] .* y[I]    
        end
        return res
    end
    #below we write y0[1] instead of y0, because sometimes the output will have a box around it, due to matrix-operations in the internal implementation
    dpdf = (x0, Fx0, y0) -> (reset_test_buf(btg); btg.debug_log = []; checkInput(x0, Fx0, y0); evalgrid_dpdf!(x0, Fx0, y0[1]); dot(grid_pdf_deriv, weightsTensorGrid))
    pdf = (x0, Fx0, y0) -> (reset_test_buf(btg); btg.debug_log = []; checkInput(x0, Fx0, y0); evalgrid_pdf!(x0, Fx0, y0[1]); dot(grid_pdf, weightsTensorGrid))
    cdf = (x0, Fx0, y0) -> (reset_test_buf(btg); btg.debug_log = []; checkInput(x0, Fx0, y0); evalgrid_cdf!(x0, Fx0, y0[1]); dot(grid_cdf, weightsTensorGrid))
    augmented_cdf_deriv = (x0, Fx0, y0) -> ( reset_test_buf(btg); reset_jac_buf(btg); derivatives == true ? (btg.debug_log = []; 
                                evalgrid_augmented_deriv!(x0, Fx0, y0[1]); 
                                #println("in augmented_cdf_deriv"); 
                                #@info "grid_augmented_deriv", size(grid_augmented_deriv);
                                #@info "grid_augmented_deriv", typeof(grid_augmented_deriv);
                                #@info "weightsTensorGrid", typeof(weightsTensorGrid);
                                dot2(grid_augmented_deriv, weightsTensorGrid)) : nothing )

    augmented_cdf_hess = (x0, Fx0, y0) -> (reset_test_buf(btg); reset_jac_buf(btg); derivatives == true ? (btg.debug_log = []; 
                                evalgrid_augmented_hess!(x0, Fx0, y0[1]); 
                                #println("in augmented_cdf_deriv"); 
                                #@info "grid_augmented_deriv", size(grid_augmented_deriv);
                                #@info "grid_augmented_deriv", typeof(grid_augmented_deriv);
                                #@info "weightsTensorGrid", typeof(weightsTensorGrid);
                                dot2(grid_augmented_hess, weightsTensorGrid)) : nothing)
    # compute estimated quantile
    # EY = (x0, Fx0) -> (evalgrid_m!(x0, Fx0); dot(grid_m, weightsTensorGrid))
    # EY2 = (x0, Fx0) -> (evalgrid_sigma_m!(x0, Fx0); dot(grid_sigma_m, weightsTensorGrid))
    # VarY =  (x0, Fx0) -> EY2(x0, Fx0) - (EY(x0, Fx0))^2
    # v = btg.trainingData.n - btg.trainingData.p
    # sigma_Y = (x0, Fx0) -> sqrt(VarY(x0, Fx0) * (v-2)/v)
    # quant_estimt = (x0, Fx0, q) -> sigma_Y(x0, Fx0) * tdistinvcdf(v, q) + EY(x0, Fx0)
    quantile_range = (x0, Fx0, q) -> (evalgrid_quantile!(x0, Fx0, q); (minimum(grid_quantile), maximum(grid_quantile)))
    quantInfo = quantile_range
    # quantInfo = (quant_estimt, EY, EY2, sigma_Y, quantile_range)
    return (pdf, cdf, dpdf, augmented_cdf_deriv, augmented_cdf_hess, quantInfo, tgridpdf, tgridcdf, tgridm, tgridsigma_m) 
end


