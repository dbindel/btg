
using LinearAlgebra

include("../datastructs.jl")
include("../kernels/kernel.jl")
include("../bayesopt/incremental.jl")
include("../quadrature/quadrature.jl")
include("../transforms/transforms.jl")


#module buffers0
#export train_buffer, test_buffer, init_train_buffer_dict, update!


######
###### Assortment of buffers for storing values in regular BTG workflow
###### Includes initialization and unpacking functions for each buffer type
######

"""
θ-dependent buffer
"""
mutable struct train_buffer
    Σθ::Array{Float64, 2}  #only top n by n block is filled
    Σθ_inv_X::Array{Float64, 2}
    qr_Σθ_inv_X::LinearAlgebra.QRCompactWY{Float64,Array{Float64,2}}
    choleskyΣθ::IncrementalCholesky
    choleskyXΣX::Cholesky{Float64,Array{Float64, 2}}
    logdetΣθ::Float64
    logdetXΣX::Float64
    capacity::Int64 #size of Σθ, maximum value of n
    n::Int64 #number data points incorporated
    k::AbstractCorrelation
    θ::Union{Array{T, 1}, T} where T<:Real
    function train_buffer(θ::Union{Array{T, 1}, T} where T<:Real, train::AbstractTrainingData, corr::AbstractCorrelation = Gaussian())
        x = train.x
        Fx = train.Fx
        n = train.n
        capacity = typeof(train)<: extensible_trainingData ? train.capacity : n #if not extensible training type, then set size of buffer to be number of data points
        Σθ = Array{Float64}(undef, capacity, capacity)
        if length(θ)>1
            Σθ[1:n, 1:n] = correlation(corr, θ, x[1:n, :]; jitter = 1e-8) #tell correlation there is single length scale
        else
            Σθ[1:n, 1:n] = correlation(corr, θ[1], x[1:n, :]; jitter = 1e-8) #tell correlation there is single length scale
        end
        choleskyΣθ = incremental_cholesky!(Σθ, n)
        Σθ_inv_X = (choleskyΣθ\Fx)
        qr_Σθ_inv_X = qr(Σθ_inv_X)
        choleskyXΣX = cholesky(Hermitian(Fx'*(Σθ_inv_X))) #regular cholesky because we don't need to extend this factorization
        new(Σθ, Σθ_inv_X, qr_Σθ_inv_X, choleskyΣθ, choleskyXΣX, logdet(choleskyΣθ), logdet(choleskyXΣX), capacity, n, corr, θ)
    end
end


"""
(θ, testingData)-dependent quantities, also depends on training data, specifically Σθ
"""
mutable struct test_buffer
    Eθ::Array{Float64, 2}
    Bθ::Array{Float64, 2}
    ΣθinvBθ::Array{Float64, 2}
    Dθ::Array{Float64, 2}
    Hθ::Array{Float64, 2}
    Cθ::Array{Float64, 2}
    test_buffer() = new()
    function test_buffer(θ::Array{Real, 1}, trainingData::AbstractTrainingData, testingData::AbstractTestingData, corr::AbstractCorrelation)::test_buffer 
        Eθ = correlation(corr, θ, testingData.x0)    
        Bθ = cross_correlation(corr, θ, testingData.x0, trainingData.x)  
        ΣθinvBθ = trainingData.choleskyΣθ\Bθ'
        Dθ = Eθ - Bθ*ΣθinvBθ
        Hθ = testingData.Fx0 - Bθ*(trainingData.Σθ_inv_X) 
        Cθ = Dθ + Hθ*(choleskyXΣX\Hθ') 
        new(Eθ, Bθ, ΣθinvBθ, Dθ, Hθ , Cθ)
    end
end
"""
Store results like Σθ_inv_y, so we have a fast way to do hat(Σθ)_inv_y, where hat(Σθ) is a submatrix
"""
mutable struct θλbuffer
    θ::Union{Array{T, 1}, T} where T<:Real
    λ::T where T<:Real
    βhat::Array{T} where T<:Real
    qtilde::Real
    Σθ_inv_y::Array{T} where T<:Real
    remainder::Array{T} where T<:Real 
    θλbuffer() = new()
    function θλbuffer(θ, λ, βhat, qtilde, Σθ_inv_y, Σθ_inv_X)
        @assert typeof(λ)<:Real
        @assert typeof(θ)<:Union{Array{T}, T} where T<: Real
        if typeof(θ)<:Array{T} where T 
            @assert length(θ) > 1 
        end
        return new(θ, λ, βhat, qtilde, Σθ_inv_y, Σθ_inv_y - Σθ_inv_X*βhat)
    end 
end

"""
Stores transformed data
"""
mutable struct λbuffer
    λ::Real
    gλz::Array{T, 1} where T<:Float64
    logjacval::Real where T<:Float64
    function λbuffer(λ, gλz, logjacval)
        return new(λ, gλz, logjacval)
    end
end

"""
Stores Σθ_inv_X_minus_i
"""
mutable struct validation_train_buffer
    θ::Union{Array{T, 1}, T} where T<:Real
    i::Int64 
    Σθ_inv_X_minus_i::Array{T} where T<:Real 
    logdetΣθ_minus_i::Float64
    logdetXΣX_minus_i::Float64
    function validation_train_buffer(θ::Union{Array{T, 1}, T} where T<:Float64, i::Int64, train_buf::train_buffer)
        (_, Σθ_inv_X, qr_Σθ_inv_X, choleskyΣθ, _, logdetΣθ, _)= unpack(train_buf) #will need QR factorization and other vals to perform fast LOOCV for least squares
        Σθ_inv_X_minus_i = lin_sys_loocv_IC(Σθ_inv_X, choleskyΣθ, i) #IC stands for incremental_cholesky 
        sublogdetΣθ = logdet_loocv(choleskyΣθ, logdetΣθ, i)
        sublogdetXΣX = logdet_XΣX_loocv(getCovariates(btg.trainingData), choleskyΣθ, logdetΣθ, i)
        return new(θ, i, Σθ_inv_X_minus_i, sublogdetΣθ, sublogdetXΣX)
    end
end

"""
Stores ΣθinvBθ
"""
mutable struct validation_test_buffer
    θ::Union{Array{T, 1}, T} where T<:Real
    ΣθinvBθ::Array{T} where T<:Real
    function validation_test_buffer(θ::Union{Array{T, 1}, T} where T<:Real, train_buf::train_buffer, test_buf::test_buffer)
        (_, _, ΣθinvBθ, _, _, _) = unpack(test_buf) #a critical assumption is that the covariates Fx0 remain constant throughout cross-validation
        (_, _, _, choleskyΣθ, _, _, _) = unpack(train_buf)
        ΣθinvBθ = lin_sys_loocv(ΣθinvBθ, U, validate) #new ΣθinvBθ of dimension n-1 x 1
        return new(θ, ΣθinvBθ)
    end
end

"""
Stores qtilde_minus_i, βhat_minus_i
"""
mutable struct validation_θλ_buffer
    θ::Union{Array{T, 1}, T} where T<:Real
    λ::Array{T, 1} where T<:Real
    i::Int64 
    βhat_minus_i  #depends on theta and lambda
    qtilde_minus_i #depends on theta and lambda
    function validation_θλ_buffer(θ::Union{Array{T, 1}, T} where T<:Float64, λ::Float64, i::Int64, train_buf::train_buffer, θλbuf::θλbuffer)
        (_, Σθ_inv_X, qr_Σθ_inv_X, choleskyΣθ, _) = unpack(train_buf) #will need QR factorization to perform fast LOOCV for least squares
        (_, _, βhat, _, _, Σθ_inv_X, remainder) = unpack(θλbuf) =  #the key here is that βhat, qtilde, etc will already have been computed if we are now doing LOOCV on model
        remainder_minus_i, βhat_minus_i = lsq_loocv(Σθ_inv_X, qr_Σθ_inv_X, remainder, βhat, i) 
        qtilde_minus_i = norm(remainder_minus_i)^2
        return new(θ, λ, i, βhat_minus_i, qtilde_minus_i)        
    end
end

"""
Stores qtilde_minus_i, βhat_minus_i
"""
mutable struct validation_λ_buffer
    logjacval::Real
    function validation_λ_buffer(λbuffer::λbuffer, i::Int64)
        (_, gλz, logjacval) = unpack(λbuffer)
        logjacval = logjacval - log(gλz[i])
        return new(logjacval)
    end
end

unpack(b::train_buffer) = (b.Σθ, b.Σθ_inv_X, b.qr_Σθ_inv_X, b.choleskyΣθ, b.choleskyXΣX, b.logdetΣθ, b.logdetXΣX)
unpack(b::test_buffer) = (b.Eθ, b.Bθ, b.ΣθinvBθ, b.Dθ, b.Hθ, b.Cθ)
unpack(b::θλbuffer) = (b.θ, b.λ, b.βhat, b.qtilde, b.Σθ_inv_y, b.remainder)
unpack(b::λbuffer) = (b.λ, b.gλz, b.logjacval)

unpack(b::validation_train_buffer) = (b.θ, b.i, b.Σθ_inv_X_minus_i, logdetΣθ_minus_i, logdetXΣX_minus_i)
unpack(b::validation_test_buffer) = (b.θ, b.ΣθinvBθ)
unpack(b::validation_θλ_buffer) = (b.θ, b.λ, b.i, b.βhat_minus_i, b.qtilde_minus_i) #depends on theta and lambda
unpack(b::validation_λ_buffer) = (b.logjacval)

#mutable struct validation_buffer_block
#    validation_test_buffer_dict::Dict{Union{Array{T}, T} where T<:Real, validation_test_buffer}
#    validation_train_buffer_dict::Dict{Union{Array{T}, T} where T<:Real, validation_train_buffer}
#    validation_θλ_buffer_dict::Dict{Tuple{Union{Array{T}, T}, T} where T<:Float64, validation_θλ_buffer}
#   function validation_buffer_block() 
#         new(Dict{Union{Array{T}, T} where T<:Real, validation_test_buffer}(), 
#         {Union{Array{T}, T} where T<:Real, validation_train_buffer}(), 
#         Dict{Tuple{Union{Array{T}, T}, T} where T<:Float64, validation_θλ_buffer}())
#    end
#end
"""
Initialize validation buffers all at once
"""
function init_validation_buffers(train_buffer_dict::Dict{Union{Array{T, 1}, T} where T<: Real, train_buffer}, θλbuffer_dict::Dict{Tuple{Union{Array{T, 1}, T}, T} where T<:Real, θλbuffer},
    test_buffer_dict::Dict{Union{Array{T, 1}, T} where T<: Real, test_buffer}, λbuffer_dict::Dict{T where T<:Real, λbuffer}, i::Int64)  
    println("Initializing validation buffers...")  
    for key in keys(train_buffer_dict)
        cur_train_buf = btg.train_buffer_dict[key]
        push!(btg.validation_train_buffer_dict, key => validation_train_buffer(key::Union{Array{T, 1}, T}, i, cur_train_buf))
    end
    for key in keys(θλbuffer_dict) #key is tuple of the form (t1, t2), where t1 is theta-nodes (Float64 if single else Array{Float64}) and t2 is lambda node (Float64)
        cur_train_buf = btg.train_buffer_dict[key]
        cur_θλbuf = btg.θλ_buffer_dict[key]
        push!(btg.validation_θλ_buffer_dict, key => validation_θλ_buffer(key[1]::Union{Array{T, 1}, T} where T<:Float64, key[2]::Float64, i, cur_train_buf::train_buffer, cur_θλbuf::θλbuffer))
    end
    for key in keys(test_buffer_dict)
        cur_train_buf = btg.train_buffer_dict[key]
        cur_test_buf = btg.test_buffer_dict[key]
        push!(btg.validation_test_buffer_dict, key => validation_test_buffer(key::Union{Array{T, 1}, T} where T<:Float64, cur_train_buf::train_buffer, cur_test_buf::test_buffer))
    end
    for key in keys(λbuffer_dict)
        cur_λbuf = btg.λbuffer_dict[key]
        push!(btg.validation_λbuffer_dict, key=> validation_λ_buffer(cur_λbuf, i::Int64))
    end
end


"""
Initialize θ to train_buffer dictionary
"""
function init_train_buffer_dict(nw::nodesWeights, trainingData::AbstractTrainingData, corr::AbstractCorrelation = Gaussian(), quadtype::String = "Gaussian")
    if getDimension(nw) == 1 #single theta length scale
        train_buffer_dict = Dict{Float64, train_buffer}()
    else
        train_buffer_dict = Dict{Array{Float64, 1}, train_buffer}()
    end
    if quadtype == "MonteCarlo"
        for i in 1:size(nw)[2]
            node = nw.nodes[:, i]
            push!(train_buffer_dict, node => train_buffer(node, trainingData, corr))
        end
    else
        #println("size nw nodes: ", size(nw.nodes))
        CI = CartesianIndices(Tuple([size(nw)[2] for i = 1:size(nw)[1]]))
        nodeSet = Set(getNodeSequence(nw.nodes, I) for I in CI)
        #println("Iterating over nodeset to build train_buffer_dict...")
        counter = 1
        for node in nodeSet #this loop is pretty expensive
            #println("Iteration: ", counter); counter += 1
            push!(train_buffer_dict, node => train_buffer(node, trainingData, corr))
        end
    end
    return train_buffer_dict 
end

"""
Initialize λbuffer_dict using nodesWeightsλ, trainingData, and nonLinearTransform 
"""
function init_λbuffer_dict(nw::nodesWeights, train::AbstractTrainingData, nt:: NonlinearTransform)
    function jac_comp(btg)
        nl2 = nw.num #number of lambda quadrature in each dimension
        Fx = getCovariates(train)
        z = getLabel(train)
        g = (x, λ) -> nt(x, λ); dg = (x, λ) -> partialx(nt, x, λ); lmbda = λ -> g(z, λ)
        gλvals = Array{Float64, 2}(undef, length(z), nl2) #preallocate space to store gλz arrays
        for i = 1:nl2
            gλvals[:, i] = lmbda(nw.nodes[i])
        end
        logjacvals = zeros(1, nl2)  #compute exponents of Jac(z)^(1-p/n)
        for i = 1:nl2
            logjacvals[i] = sum(log.(abs.(map( x-> dg(x, nw.nodes[i]), z))))
        end
        return gλvals, logjacvals
    end
    gλvals, logjacvals = jac_comp(btg)
    λbuffer_dict = Dict{Real, λbuffer}()
    for i = 1:size(gλvals, 2)
        cur_node = nw.nodes[i]
        @assert typeof(cur_node) <: Real #lambda is always real
        push!(λbuffer_dict, cur_node => λbuffer(cur_node, gλvals[:, i], logjacvals[i]))
    end
    return λbuffer_dict
end

"""
Create dictionary of qtilde, βhat, and Σθ_inv_y values.
These are essentially the minimizer and minimum of least squares problem involving the transformed observations, covariance matrix, and covariates
Must be initialized after λbuffer_dict and train_buffer_dict. This function is only called once in the lifetime of the btg object, namely when it is initialized.
"""
function init_θλbuffer_dict(nwθ::nodesWeights, nwλ::nodesWeights, train::AbstractTrainingData, λbuffer_dict::Union{Dict{T, λbuffer}, Dict{Array{T, 1}, λbuffer}} where T<:Real, train_buffer_dict::Union{Dict{Array{T, 1}, train_buffer}, Dict{T, train_buffer}}  where T<: Real, quadtype::Array{String,1})
    
    #println("type of lambda buffer dict: ", typeof(λbuffer_dict))
    #λbuffer_dict::Dict{Union{Array{T, 1}, T}, λbuffer} where T<:Real
    (R, _) = get_btg_iterator(nwθ, nwλ, quadtype)
    d = getDimension(nwθ)
    if d==1
        θλbuffer_dict = Dict{Tuple{T, T} where T<:Real, θλbuffer}()   
    else
        θλbuffer_dict = Dict{Tuple{Array{T, 1}, T} where T<:Real, θλbuffer}()   
    end
    for I in R
        (r1, r2, t1, t2) = get_index_slices(nwθ::nodesWeights, nwλ::nodesWeights, quadtype::Array{String,1}, I) #t1, t2 are keys for theta and lambda, respectively
        @assert typeof(t1)<:Union{T, Array{T, 1}} where T<:Real
        @assert typeof(t2)<:Real
        cur_train_buf = train_buffer_dict[t1] #get train_buffer
        cur_λ_buf = λbuffer_dict[t2] #get lambda buffer
        θλpair = (t1, t2)::Union{Tuple{Array{T, 1}, T}, Tuple{T, T}} where T<:Real #key used for this buffer
        (_, Σθ_inv_X, qr_Σθ_inv_X, choleskyΣθ, choleskyXΣX) = unpack(cur_train_buf)
        (λ, gλz, logjacval) = unpack(cur_λ_buf)
        choleskyXΣX = cur_train_buf.choleskyXΣX
        choleskyΣθ = cur_train_buf.choleskyΣθ
        Fx = getCovariates(train)
        Σθ_inv_y = (choleskyΣθ \ gλz) #O(n^2)
        βhat = choleskyXΣX\(Fx'*Σθ_inv_y)  #O
        #qtilde = (expr = gλz-Fx*βhat; expr'*(choleskyΣθ\expr))
        qtilde =  gλz'*Σθ_inv_y  - 2*gλz'*Σθ_inv_X*βhat + βhat'*Fx'*Σθ_inv_X*βhat #O(np)
        cur_θλbuffer = θλbuffer(t1, t2, βhat, qtilde, Σθ_inv_y, Σθ_inv_X)
        push!(θλbuffer_dict, θλpair => cur_θλbuffer)
    end
        return θλbuffer_dict
end

"""
Initializes test_buffer_dict with empty test buffers, so that the keys match those of train_buffer
"""
function init_test_buffer_dict(nw::nodesWeights, train_buffer_dict::Union{Dict{Array{T, 1}, train_buffer}, Dict{T, train_buffer}} where T<: Real)
    d = getDimension(nw)
    if d==1 #if single length scale, then key will be Real, else array
        return Dict{Real, test_buffer}(arr => test_buffer() for arr in keys(train_buffer_dict))
    else
        return Dict{Array{T, 1} where T<:Real, test_buffer}(arr => test_buffer() for arr in keys(train_buffer_dict))
    end
end

function init_validation_train_buffer_dict(nw::nodesWeights)
    d = getDimension(nw)
    if d==1
        return Dict{T where T<: Real, validation_train_buffer}() 
    else
        return Dict{Array{T, 1} where T<: Real, validation_train_buffer}() 
    end
end

function init_validation_θλ_buffer_dict(nw::nodesWeights)
    d = getDimension(nw)
    if d==1
        return Dict{Tuple{T, T} where T<: Real, validation_θλ_buffer}() 
    else
        return Dict{Tuple{Array{T, 1}, T} where T<: Real, validation_θλ_buffer}() 
    end
end

function init_validation_test_buffer_dict(nw::nodesWeights)
    d = getDimension(nw)
    if d==1
        return Dict{T where T<: Real, validation_test_buffer}() 
    else
        return Dict{Array{T, 1} where T<: Real, validation_test_buffer}() 
    end
end

function init_validation_λ_buffer_dict()
    return Dict{T where T<: Real, validation_λ_buffer}()
end

######
###### Update train_buffer when extending kernel system (for Bayesian optimization)
###### Update test_buffer for each new problem, parametrized by triple (x0, Fx0, y0)
######


"""
Updates training buffer with testing buffer
NOTE: trainingData field of btg must be updated before train_buffer is updated
"""
function update!(train_buffer::train_buffer, test_buffer::test_buffer, trainingData::AbstractTrainingData)  #use incremental cholesky to update training train_buffer
    @assert typeof(x0)<:Array{T, 2} where T<:Real
    @assert typeof(Fx0)<:Array{T, 2} where T<:Real
    @assert typeof(y0)<:Array{T, 1} where T<:Real 
    @assert train_buffer.n < trainingData.n #train_train_buffer must be "older" than trainingData
    k = trainingData.n - train_buffer.n 
    A12, A2 = extend(train_buffer.choleskyΣθ, k) #two view objects 
    #A12 = cross_correlation(train_buffer.k(), train_buffer.θ, trainingData.x[1:end-k], trainingData.x[end-k+1:end])
    #A2 = correlation(train_buffer.k(), train_buffer.θ, trainingData.x[end-k+1:end]) #Σθ should get updated automatically, but only upper triangular portion
    A12 = test_buffer.Bθ'
    A2 = test_buffer.Eθ
    update!(train_buffer.choleskyΣθ, k) #extends Cholesky decomposition
    train_buffer.Σθ_inv_X = train_buffer.choleskyΣθ\trainingData.Fx
    train_buffer.n += k
    return nothing
end

"""
Update test_buffer, which depends on testing data, training data, and train_buffer. 
"""
function update!(train_buffer::train_buffer, test_buffer::test_buffer, trainingData::AbstractTrainingData, testingData::AbstractTestingData)
    @assert checkCompatibility(trainingData, testingData) #make sure testingData is compatible with trainingData
    if length(train_buffer.θ)==1 #check if θ is an array of length 1
        θ = train_buffer.θ[1] #unbox θ, so correlation knows to use a single length scale 
    else 
        θ = train_buffer.θ
    end
    test_buffer.Eθ = correlation(train_buffer.k, θ, testingData.x0)    
    test_buffer.Bθ = cross_correlation(train_buffer.k, θ, testingData.x0, trainingData.x)  
    test_buffer.ΣθinvBθ = train_buffer.choleskyΣθ\test_buffer.Bθ'
    test_buffer.Dθ = test_buffer.Eθ - test_buffer.Bθ*test_buffer.ΣθinvBθ
    #println("shape of Dtheta: ", size(test_buffer.Dθ ))
    #println("shape of Btheta: ", size(test_buffer.Bθ ))
    #println("shape of Fx0: " , size(testingData.Fx0))
    #println("shape of train_buffer.Σθ_inv_X", size(train_buffer.Σθ_inv_X))
    test_buffer.Hθ = testingData.Fx0 - test_buffer.Bθ*(train_buffer.Σθ_inv_X) 
    test_buffer.Cθ = test_buffer.Dθ + test_buffer.Hθ*(train_buffer.choleskyXΣX\test_buffer.Hθ') 
    return nothing
end



