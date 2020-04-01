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
    quadType::String #Gaussian, Turan, or MonteCarlo
    priorθ::priorType
    priorλ::priorType
    nodesWeightsθ #integration nodes and weights for θ
    nodesWeightsλ #integration nodes and weights for λ; nodes and weights should remain constant throughout the lifetime of the btg object
    train_buffer_dict::Dict{Float64, train_buffer}  #buffer for each theta value
    test_buffer_dict::Dict{Float64, test_buffer} #buffer for each theta value
    capacity::Int64
    btg(train_data::AbstractTrainingData, rangeθ, rangeλ; corr::AbstractCorrelation = Gaussian(), priorθ::priorType = Uniform(rangeθ), priorλ::priorType = Uniform(rangeλ), quadtype::String = "Gaussian", transform:NonlinearTransform = BoxCox())
        #a btg object really should contain a bunch of train buffers correpsonding to different theta-values
        #we should add some fields to the nodesweights_theta data structure to figure out the number of dimensions we are integrating over...should we allow different length scale ranges w/ different quadrature nodes? I think so??
        nodesWeightsθ = nodesWeights(rangeθ, quadtype)
        nodesWeightsλ = nodesWeights(rangeλ, quadtype)
        train_buffer_dict  = init_train_buffer_dict(nodesWeightsθ, training_data, corr)
        test_buffer_dict = Dict{Array{Real, 1}, test_buffer}() #empty dict
        new(train_data, 0, transform, corr, quadtype, priorθ, priorλ, nodesWeightsθ, nodesWeightsλ, train_buffer_dict, test_buffer_dict, getCapacity(train_data))
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
    update!(btg.train_data, x0, Fx0, y0)    
    update_train_buffer!(btg.train_buffer, btg.train)
end

function solve(btg::btg)
    WeightTensorGrid = weight_comp(btg::btg)
    pdf, cdf, pdf_deriv = prediction_comp(btg, WeightTensorGrid)
end

"""
Compute weights in the mixture of T-distributions
"""
function weight_comp(btg::BTG)#depends on train_data and not test_data
    # line 36-99 in tensorgrid.jl 
    nd = btg.nodesWeightsθ.d + 1 #number of dimensions of theta (d) plus number dimensions of lambda (1)
    nq = btg.nodesWeightsθ.num #number of quadrature nodes
    n =  btg.train_data.n; p = btg.train_data.p #number of training points and dimension of covariates
    weightsTensorGrid = Array{Float64, nd}(undef, Tuple([nq for i = 1:nd])) #initialize tensor grid
    R = CartesianIndices(weightsTensorGrid)
    for I in R #I is multi-index
        weightsTensorGrid[I] = getProd(vcat(btg.nodesWeightsθ.weights, btg.nodesWeightsλ.weights) #this step can be simplified because the tensor is symmetric (weights are the same along each dimension)
    end

    #compute exponents of qtilde^(-(n-p)/2)
    qTensorGrid = similar(weightsTensorGrid)
    z = getLabel(btg.train_data)
    g = (x, λ) -> btg.g(x, λ); dg = (x, λ) -> partialx(btg.g, x, λ)
    lmbda = λ -> g(z, λ)
    gλvals = Array{Float64, 2}(undef, length(z), nq) #preallocate space to store gλz arrays
    for i = 1:nq 
        gλvals[:, i] = lmbda(btg.nodesWeightsλ.nodes[i])
    end
    Fx = getCovariates(btg.train_data)
    for I in R
        train_buffer = btg.train_buffer_dict[getNodeSequence(Tuple(I)[1:end-1])] #look up train buffer based on combination of theta quadrature nodes
        choleskyXΣX = train_buffer.choleskyXΣX
        choleskyΣθ = train_buffer.choleskyΣθ
        gλz = gλvals[:, Tuple(I)[end]]
        βhat = choleskyXΣX\(Fx'*(choleskyΣθ\gλz)) 
        qtilde = (expr = gλz-Fx*βhat; expr'*(choleskyΣθ\expr))
        qTensorGrid(I) = -(n-p)/2 * log(qtilde)
    end

    #compute exponents of |Σθ|^(-1/2) and |X'ΣθX|^(-1/2)  
    detTensorGridΣθ = similar(weightsTensorGrid)
    detTensorGridXΣX = similar(weightsTensorGrid) 
    for I in R
        train_buffer = btg.train_buffer_dict[getNodeSequence(Tuple(I)[1:end-1])] #look up train buffer based on combination of theta quadrature nodes
        choleskyΣθ = train_buffer.choleskyΣθ
        choleskyXΣX = train_buffer.choleskyXΣX
        detTensorGridΣθ[I] = -0.5 * logdet((choleskyΣθ)) #log determinant of incremental cholesky
        detTensorGridXΣX[I] = -1.0 * sum(log.(diag(choleskyXΣX.U))) 
    end
    
    #compute exponents of (Jac(z)^(1-p/n)
    jacTensorGrid = similar(weightsTensorGrid)
    jacvals = zeros(1, nq)
    for i = 1:nq
        jacvals[i] = sum(log.(abs.(map( x-> dg(x, btg.nodesWeightsλ.nodes[i]), z))))
    end
    for I in R
        jacTensorGrid[I] = (1-p/n) * jacvals[Tuple(I)[end]]
    end

    #compute exponents of pθ(θ)*pλ(λ)
    for I in R
        
    end

    @assert size(qTensorGrid) == size(detTensorGridΣθ) == size(detTensorGridXΣX) == size(jacTensorGrid) == size(priorTensorGrid)
    powerGrid =  qTensorGrid + detTensorGrid + jacTensorGrid #sum of exponents
    powerGrid = exp.(powerGrid .- maximum(powerGrid)) #linear scaling
    weightsTensorGrid = powerGrid .* weightsTensorGrid
    weightsTensorGrid =  weightsTensorGrid/sum(weightsTensorGrid) #normalized grid of weights
    return WeightTensorGrid
end

"""
Compute pdf and cdf functions
"""
function prediction_comp(btg::BTG, weightTensorGrid::Array{Float64}) #depends on both train_data and test_data
    update_test_buffer!(train_buffer::train_buffer, test_buffer::test_buffer, trainingData::AbstractTrainingData, testingData::AbstractTrainingData)
    
    tgridpdf = similar(weightsTensorGrid) 
    #tensor grid for CDF P(z0|theta, lambda, z)
    tgridcdf = similar(weightsTensorGrid)
    R = CartesianIndex(weightsTensorGrid)
    for I in R
        tgridpdf[I] = 
        tgridcdf[I] =  
    end

    for i = 1:l1
        for j = 1:l2 
            funcs = integrand(nodesθ[i], nodesλ[j], train, test, boxCoxObj, theta_param_list[i], quadtype)
            for k = 1:l3
                #tgrid[i, j, k] = funcs[k]       
                tgridpdf[i, j, k] = funcs[1][k]
                tgridcdf[i, j, k] = funcs[2][k]
            end
        end
    end
    function evalTgrid_pdf(z0)
        res = Array{Float64, 3}(undef, l1, l2, l3)
        for i=1:l1
            for j = 1:l2
                for k =1:l3
                    #println("tgridpdf(z0): ", tgridpdf[i, j, k](z0))
                    res[i, j, k] = tgridpdf[i, j, k](x0, Fx0, y0)[1]
                end
            end
        end 
        return res
    end
    function evalTgrid_cdf(z0)
        res = Array{Float64, 3}(undef, l1, l2, l3)
        for i=1:l1
            for j = 1:l2
                for k =1:l3
                    res[i, j, k] = tgridcdf[i, j, k](x0, Fx0, y0)
                end
            end
        end 
        return res
    end
    #product rule loop for Turan quadrature (k>1). In the k=1 case, this reduces to taking the dot
    # product between the tensor grid and weights grid.
    function pdf(x0, Fx0, y0)
        grid = evalTgrid_pdf(z0)
        res = 0
        for i = 1:n3
            for j = 1:i
                res = binomial(i, j) * dot(grid[:, :, j],  weightsTensorGrid[:, :, n3-j+1])  
            end
        end
        return res
    end
    function cdf(x0, Fx0, y0)
        grid = evalTgrid_cdf(z0)
        res = 0
        for i = 1:n3
            for j = 1:i
                res = binomial(i, j) * dot(grid[:, :, j], weightsTensorGrid[:, :, n3-j+1])  
            end
        end
        return res
    end
    return (pdf, cdf)
    
    
    # line 100-159 in tensorgrid.jl
    return pdf, cdf, pdf_deriv
end

