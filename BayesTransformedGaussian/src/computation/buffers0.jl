
using LinearAlgebra

include("../datastructs.jl")
include("../kernels/kernel.jl")
include("../bayesopt/incremental.jl")
include("../quadrature/quadrature.jl")
include("../transforms/transforms.jl")

using TimerOutputs
#module buffers0
#export train_buffer, test_buffer, init_train_buffer_dict, update!
if !@isdefined(to)
    const to = TimerOutput()
end

######s
###### Assortment of buffers for storing values in regular BTG workflow
###### Includes initialization and unpacking functions for each buffer type
######

"""
θ-dependent buffer
"""
mutable struct train_buffer
    Σθ::Array{Float64, 2}  #only top n by n block is filled
    Σθ_inv_X::Array{Float64, 2}
    qr_Σθ_inv_X::Union{LinearAlgebra.QRCompactWY{Float64,Array{Float64,2}}, Nothing}
    choleskyΣθ::IncrementalCholesky
    choleskyXΣX::Cholesky{Float64,Array{Float64, 2}}
    logdetΣθ::Float64
    logdetXΣX::Float64
    capacity::Int64 #size of Σθ, maximum value of n
    n::Int64 #number data points incorporated
    k::AbstractCorrelation
    θ::Union{Array{T, 1}, T} where T<:Real
    L_inv_X::Union{Array{Float64, 2}, Nothing}
    function train_buffer(θ::Union{Array{T, 1}, T} where T<:Real, train::AbstractTrainingData, corr::AbstractCorrelation = Gaussian())
        #x = train.x
        #Fx = train.Fx
        #n = train.n
        x = getPosition(train)
        Fx = getCovariates(train)
        n = getNumPts(train)
        capacity = typeof(train)<: extensible_trainingData ? train.capacity : n #if not extensible training type, then set size of buffer to be number of data points
        @timeit to "Σθ" begin
            Σθ = Array{Float64}(undef, capacity, capacity)
            if length(θ)>1
                Σθ[1:n, 1:n] = correlation(corr, θ, x[1:n, :]; jitter = 1e-12) #tell correlation there is single length scale
            else
                Σθ[1:n, 1:n] = correlation(corr, θ[1], x[1:n, :]; jitter = 1e-12) #tell correlation there is single length scale
            end
            choleskyΣθ = incremental_cholesky!(Σθ, n)
        end
        #L = get_chol(choleskyΣθ).L
        #U = get_chol(choleskyΣθ).U
        #L_inv_X = L\Fx
        L_inv_X = nothing #not used for now
        #qr_Σθ_inv_X = qr(L_inv_X)
        qr_Σθ_inv_X = nothing #we don't use the qr factorization right now
        Σθ_inv_X = choleskyΣθ\Fx
        #@info "Fx'*(Σθ_inv_X)", Fx'*(Σθ_inv_X)
        #try

        XΣ_inv_X = Hermitian(Fx'*(Σθ_inv_X))
        #@info minimum(eigvals(XΣ_inv_X))
        @assert issymmetric(XΣ_inv_X)
        @assert isposdef(XΣ_inv_X)
        choleskyXΣX = cholesky(XΣ_inv_X) #regular cholesky because we don't need to extend this factorization
        #catch e
        #    @info "Hermitian(Fx'*(Σθ_inv_X))", Hermitian(Fx'*(Σθ_inv_X))
        #end
        new(Σθ, Σθ_inv_X, qr_Σθ_inv_X, choleskyΣθ, choleskyXΣX, logdet(choleskyΣθ), logdet(choleskyXΣX), capacity, n, corr, θ, L_inv_X)
    end
end

"""
(θ, testingData)-dependent quantities, also depends on training data, specifically Σθ
"""
mutable struct test_buffer 
    Eθ::Union{Array{Float64, 2}, Nothing}
    Bθ::Union{Array{Float64, 2}, Nothing}
    ΣθinvBθ::Union{Array{Float64, 2}, Nothing}
    Dθ::Union{Array{Float64, 2}, Nothing}
    Hθ::Union{Array{Float64, 2}, Nothing}
    Cθ::Union{Array{Float64, 2}, Nothing}
    θ::Union{Union{Array{T, 1}, T}, Nothing} where T<:Real
    update_bit::Union{Bool, Nothing} #can we update the test buffer without recomputing stuff?
    init_bit::Bool #has the test_buffer been initialized yet?
    test_buffer() = new(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, false)
end

mutable struct jac_buffer
    jacB::Union{Union{Array{T, 1}, Array{T, 2}}, Nothing} where T<:Float64
    jacD::Union{Union{Array{T, 1}, Array{T, 2}}, Nothing} where T<:Float64
    jacH::Union{Union{Array{T, 1}, Array{T, 2}}, Nothing} where T<:Float64
    jacC::Union{Union{Array{T, 1}, Array{T, 2}}, Nothing} where T<:Float64
    update_bit::Union{Bool, Nothing} #can we update the test buffer without recomputing stuff?
    init_bit::Bool #has the test_buffer been initialized yet?
    jac_buffer() = new(nothing, nothing, nothing, nothing, nothing, false)
end

"""
Store results like Σθ_inv_y, so we have a fast way to do hat(Σθ)_inv_y, where hat(Σθ) is a submatrix
"""
mutable struct θλbuffer
    θ::Union{Array{T, 1}, T} where T<:Real
    λ::T where T<:Real
    βhat::Array{T} where T<:Real
    qtilde::Real
    L_inv_y::Union{Array{T} where T<:Real, Nothing}
    Σθ_inv_y::Array{T} where T<:Real 
    remainder::Union{Array{T} where T<:Real, Nothing}
    Σθ_inv_remainder::Array{T} where T<:Real
    θλbuffer() = new()
    function θλbuffer(θ, λ, βhat, qtilde, L_inv_y, Σθ_inv_y, remainder, Σθ_inv_remainder)
        @assert typeof(λ)<:Real
        @assert typeof(θ)<:Union{Array{T}, T} where T<: Real
        if typeof(θ)<:Array{T} where T 
            @assert length(θ) > 1 
        end
        return new(θ, λ, βhat, qtilde, L_inv_y, Σθ_inv_y, remainder, Σθ_inv_remainder) #Σθ_inv_y - Σθ_inv_X*βhat)
    end 
end



"""
Stores transformed data
"""
mutable struct λbuffer
    λ::Real
    gλz::Array{T, 1} where T<:Float64
    logjacval::Real where T<:Float64
    dgλz::Array{T, 1} where T<:Float64
    function λbuffer(λ, gλz, logjacval, dgλz)
        return new(λ, gλz, logjacval, dgλz)
    end
end



"""
update λbuffer with single new y-observation
"""
function update!(λbuffer::λbuffer, y_new::Union{A, B} where A<:Real where B<:Array{H, 1} where H<:Real, g::NonlinearTransform)
    if typeof(y_new)<:Array{T, 1} where T<:Real && length(y_new)==1
        y_new = y_new[1]
    end
    λ = λbuffer.λ;
    #@info "partialx", typeof(partialx)
    dg = x-> partialx(g, x, λ)   #fix lambda
    push!(λbuffer.gλz, g(y_new, λ))
    dgy_new = dg(y_new)
    λbuffer.logjacval = λbuffer.logjacval + log(abs(dgy_new))
    push!(λbuffer.dgλz, dgy_new)
end

"""
Updates θλbuffer
Must have already updated train_buffer, lambda_buffer, and train_data before calling this function
"""
function update!(θλbuffer::θλbuffer, cur_λbuffer::λbuffer, train_buffer::train_buffer, train_data::AbstractTrainingData)
    #(_, Σθ_inv_X, qr_Σθ_inv_X, choleskyΣθ, choleskyXΣX) = unpack(train_buffer)
    #L_inv_X = cur_train_buf.L_inv_X
    (λ, gλz, logjacval) = unpack(cur_λbuffer)
    Σθ_inv_X = train_buffer.Σθ_inv_X
    choleskyXΣX = train_buffer.choleskyXΣX
    choleskyΣθ = train_buffer.choleskyΣθ
    Fx = getCovariates(train_data)
    @timeit to "Σθ_inv_y" Σθ_inv_y = choleskyΣθ\gλz
    #Σθ_inv_y = (choleskyΣθ \ gλz) #O(n^2)
    @timeit to "βhat" βhat = choleskyXΣX\(Fx'*Σθ_inv_y)  #O
    #qtilde = (expr = gλz-Fx*βhat; expr'*(choleskyΣθ\expr))
    @timeit to "qtilde" qtilde =  gλz'*Σθ_inv_y  - 2*gλz'*Σθ_inv_X*βhat + βhat'*Fx'*Σθ_inv_X*βhat #O(np) checks out b/c qtilde = norm(remainder)^2
    #remainder = L_inv_y - L_inv_X*βhat
    θλbuffer.qtilde = qtilde
    θλbuffer.βhat = βhat
    θλbuffer.Σθ_inv_y = Σθ_inv_y
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
    choleskyXΣX_X_minus_i::Array{Float64, 2}
    function validation_train_buffer(θ::Union{Array{T, 1}, T} where T<:Float64, i::Int64, train_buf::train_buffer, trainingData::AbstractTrainingData)
        (_, Σθ_inv_X, qr_Σθ_inv_X, choleskyΣθ, _, logdetΣθ, _)= unpack(train_buf) #will need QR factorization and other vals to perform fast LOOCV for least squares
        Σθ_inv_X_minus_i = lin_sys_loocv_IC(Σθ_inv_X, choleskyΣθ, i) #IC stands for incremental_cholesky 
        Fx_minus_i = getCovariates(trainingData)[[1:i-1;i+1:end], :]
        sublogdetΣθ = logdet_loocv_IC(choleskyΣθ, logdetΣθ, i)
        sublogdetXΣX = logdet_XΣX_loocv_IC(getCovariates(trainingData), choleskyΣθ, logdetΣθ, i)
        choleskyXΣX_X_minus_i = cholesky(Hermitian(Fx_minus_i'*(Σθ_inv_X_minus_i))) \ (Fx_minus_i')  # for computing beta(-i)
        return new(θ, i, Σθ_inv_X_minus_i, sublogdetΣθ, sublogdetXΣX, choleskyXΣX_X_minus_i)
    end
end

"""
Stores ΣθinvBθ
"""
mutable struct validation_test_buffer
    θ::Union{Array{T, 1}, T} where T<:Real
    ΣθinvBθ::Array{T} where T<:Real
    validation_test_buffer() = new()
    #function validation_test_buffer(θ::Union{Array{T, 1}, T} where T<:Real, train_buf::train_buffer, test_buf::test_buffer)
    #    #ΣθinvBθ = test_buf.ΣθinvBθ
    #    #choleskyΣθ = train_buf.choleskyΣθ 
    #    # now these are giving me issues vvv
    #    (_, _, ΣθinvBθ, _, _, _) = unpack(test_buf) #a critical assumption is that the covariates Fx0 remain constant throughout cross-validation
    #    (_, _, _, choleskyΣθ, _, _, _) = unpack(train_buf)
    #    ΣθinvBθ = lin_sys_loocv_IC(ΣθinvBθ, choleskyΣθ, validate) #new ΣθinvBθ of dimension n-1 x 1
    #    return new(θ, ΣθinvBθ)
    #end
end



"""
Stores qtilde_minus_i, βhat_minus_i
"""
mutable struct validation_θλ_buffer
    θ::Union{Array{T, 1}, T} where T<:Real
    λ::Real
    i::Int64 
    βhat_minus_i::Array{T} where T<:Real  #depends on theta and lambda
    qtilde_minus_i::Real #depends on theta and lambda
    Σθ_inv_y_minus_i::Array{T} where T<:Real
    # remainder_minus_i
    function validation_θλ_buffer(θ::Union{Array{T, 1}, T} where T<:Float64, λ::Float64, i::Int64, train_buf::train_buffer, λbuf::λbuffer, θλbuf::θλbuffer, val_train_buf::validation_train_buffer, trainingData::AbstractTrainingData)
        # why can't this line below use/find unpack?
        #println(" type of unpacked train_buf in buffers.jl 157:", typeof(unpack(train_buf)))
        #(_, Σθ_inv_X, qr_Σθ_inv_X, choleskyΣθ, _) = unpack(train_buf) #will need QR factorization to perform fast LOOCV for least squares
        Σθ_inv_X =  train_buf.Σθ_inv_X
        # qr_Σθ_inv_X = train_buf.qr_Σθ_inv_X 
        choleskyΣθ = train_buf.choleskyΣθ
        # L_inv_X = train_buf.L_inv_X
        #Σθ_inv_X = train_buf.Σθ_inv_X; qr_Σθ_inv_X = train_buf.qr_Σθ_inv_X; choleskyΣθ = train_buf.choleskyΣθ
        #println(" type of unpacked theta_lambda buf in buffers.jl 157:", typeof(unpack(θλbuf)))
        #(_, _, βhat, _, Σθ_inv_y, remainder, Σθ_inv_remainder) = unpack(θλbuf)  #the key here is that βhat, qtilde, etc will already have been computed if we are now doing LOOCV on model
        # βhat = θλbuf.βhat
        Σθ_inv_y = θλbuf.Σθ_inv_y
        # remainder = θλbuf.remainder
        # Σθ_inv_remainder = θλbuf.Σθ_inv_remainder
        Σθ_inv_y_minus_i = lin_sys_loocv_IC(Σθ_inv_y, choleskyΣθ, i)
        Σθ_inv_X_minus_i = lin_sys_loocv_IC(Σθ_inv_X, choleskyΣθ, i)
        gλz_minus_i = λbuf.gλz[[1:i-1;i+1:end]]
        Fx_minus_i = trainingData.Fx[[1:i-1;i+1:end], :]
        # remainder_minus_i, βhat_minus_i = lsq_loocv(L_inv_X, qr_Σθ_inv_X, remainder, βhat, i) 
        # qtilde_minus_i = norm(remainder_minus_i[[1:i-1;i+1:end]])^2
        choleskyXΣX_X_minus_i = val_train_buf.choleskyXΣX_X_minus_i
        βhat_minus_i = choleskyXΣX_X_minus_i * Σθ_inv_y_minus_i  
        temp = Σθ_inv_y_minus_i - Σθ_inv_X_minus_i*βhat_minus_i
        qtilde_minus_i = ((gλz_minus_i - Fx_minus_i*βhat_minus_i)' * temp)[1]
        #qtilde_minus_i = norm(remainder_minus_i)^2 very blatantly wrong
        #Σθ_inv_remainder_minus_i = lin_sys_loocv_IC(Σθ_inv_remainder, choleskyΣθ, i) 
        #qtilde_minus_i = remainder_minus_i[[1:i-1;i+1:end]]'*Σθ_inv_remainder_minus_i
        #@info "Σθ_inv_remainder_minus_i" Σθ_inv_remainder_minus_i
        #@info "remainder_minus_i[[1:i-1;i+1:end]]" remainder_minus_i[[1:i-1;i+1:end]]
        #@info "alternate comp of Σθ_inv_remainder_minus_i" choleskyΣθ\remainder_minus_i
        # @info "qtilde_minus_i" qtilde_minus_i
        @assert typeof(qtilde_minus_i)<:Real
        return new(θ, λ, i, βhat_minus_i, qtilde_minus_i, Σθ_inv_y_minus_i)        
    end
end

"""
Stores log jacobian of gλz with ith point deleted. i is not stored because the i is specific
to the call to solve. Rather it's used to construct a validation_λ_buffer.
"""
mutable struct validation_λ_buffer
    logjacval::Real
    function validation_λ_buffer(λbuffer::λbuffer, i::Int64)
        (_, _, logjacval, dgλz) = unpack(λbuffer)
        logjacval = logjacval - log(abs(dgλz[i]))
        return new(logjacval)
    end
end

function print_test_buffer(b::test_buffer)
    println("Eθ: ", b.Eθ)
    println("Bθ: ", b.Bθ)
    println("ΣθinvBθ: ", b.ΣθinvBθ)
    println("Dθ: ", b.Dθ)
    println("Hθ: ", b.Hθ)
    println("Cθ: ", b.Cθ)
end


"""
Checks whether a θλbuffer buffer corresponds to a validation_θλ_buffer
"""
function equals(buf::θλbuffer, val_buf::validation_θλ_buffer)
    try 
        @assert buf.θ == val_buf.θ 
        @assert buf.λ == val_buf.λ
        @assert isapprox(buf.βhat, val_buf.βhat_minus_i)
        @assert isapprox(buf.qtilde, val_buf.qtilde_minus_i)
    catch e
        return false
    end
    return true
end
"""
Checks if a θλbuffer_dict is equal to a validation_θ_λ_buffer_dict
"""
function equals(θλbuffer_dict::Union{Dict{Tuple{Array{T, 1}, T} where T<:Real , θλbuffer}, Dict{Tuple{T, T} where T<:Real , θλbuffer}}, val_θλbuffer_dict::Union{Dict{Tuple{Array{T, 1}, T} where T<:Real, validation_θλ_buffer}, Dict{Tuple{T, T} where T<:Real, validation_θλ_buffer}})
        @assert keys(θλbuffer_dict) == keys(val_θλbuffer_dict)
        for key in keys(θλbuffer_dict)
            try
                @assert equals(θλbuffer_dict[key], val_θλbuffer_dict[key])
            catch e
                @warn "bad key", key 
                return false
            end
        end
    return true
end

###########   what's going on here???? ##############################
function anotherone(b::test_buffer)
    #println("in anotherone")
    return (b.Eθ, b.Bθ, b.ΣθinvBθ, b.Dθ, b.Hθ, b.Cθ)
end
##################### I now deem the unpack functions to be unsafe...it's hard to count underscores...
##################### to get a field, just use dot notation

unpack(b::θλbuffer) = (b.θ, b.λ, b.βhat, b.qtilde, b.L_inv_y, b.Σθ_inv_y, b.remainder, b.Σθ_inv_remainder)
function anotherone(b::θλbuffer)
    return (b.θ, b.λ, b.βhat, b.qtilde, b.L_inv_y, b.Σθ_inv_y, b.remainder, b.Σθ_inv_remainder)
end

function unpack(b::test_buffer) 
    #println("in unpack")
    return (b.Eθ, b.Bθ, b.ΣθinvBθ, b.Dθ, b.Hθ, b.Cθ, b.θ)
end
unpack(b::train_buffer) = (b.Σθ, b.Σθ_inv_X, b.qr_Σθ_inv_X, b.choleskyΣθ, b.choleskyXΣX, b.logdetΣθ, b.logdetXΣX, b.θ)
unpack(b::λbuffer) = (b.λ, b.gλz, b.logjacval, b.dgλz)
unpack(b::validation_train_buffer) = (b.θ, b.i, b.Σθ_inv_X_minus_i, b.logdetΣθ_minus_i, b.logdetXΣX_minus_i)
unpack(b::validation_test_buffer) = (b.θ, b.ΣθinvBθ)
unpack(b::validation_θλ_buffer) = (b.θ, b.λ, b.i, b.βhat_minus_i, b.qtilde_minus_i, b.Σθ_inv_y_minus_i) #depends on theta and lambda
unpack(b::validation_λ_buffer) = (b.logjacval)
unpack(b::jac_buffer) = (b.jacC, b.jacB, b.jacH, b.jacD)

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
Initialize θ to train_buffer dictionary
"""
function init_train_buffer_dict(nw::nodesWeights, trainingData::AbstractTrainingData, corr::AbstractCorrelation = Gaussian(), quadtype::String = "Gaussian")
    if getDimension(nw) == 1 #single theta length scale
        train_buffer_dict = Dict{Float64, train_buffer}()
        for i in 1:size(nw)[2]
            node = nw.nodes[:, i][1]
            push!(train_buffer_dict, node => train_buffer(node, trainingData, corr))
        end
    else
        train_buffer_dict = Dict{Array{Float64, 1}, train_buffer}()
        if quadtype != "Gaussian" # for MC, QMC, SparseGrid
            for i in 1:size(nw)[2]
                node = nw.nodes[:, i]
                push!(train_buffer_dict, node => train_buffer(node, trainingData, corr))
            end
        else
            #println("size nw nodes: ", size(nw.nodes))
            CI = CartesianIndices(Tuple([size(nw)[2] for i = 1:size(nw)[1]]))
            nodeSet = Set(getNodeSequence(nw.nodes, I) for I in CI)
            #println("Iterating over nodeset to build train_buffer_dict...")
            # counter = 1
            for node in nodeSet #this loop is pretty expensive
                #println("Iteration: ", counter); counter += 1
                push!(train_buffer_dict, node => train_buffer(node, trainingData, corr))
            end
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
        g = (x, λ) -> nt(x, λ); dg = (x, λ) -> partialx(nt, x, λ); lmbda = λ -> g(z, λ); prime =  λ -> dg(z, λ) 
        gλvals = Array{Float64, 2}(undef, length(z), nl2) #preallocate space to store gλz arrays
        dgλvals = Array{Float64, 2}(undef, length(z), nl2)
        for i = 1:nl2
            gλvals[:, i] = lmbda(nw.nodes[i])
            dgλvals[:, i] .= prime(nw.nodes[i])
        end
        logjacvals = zeros(1, nl2)  #compute exponents of Jac(z)^(1-p/n)
        for i = 1:nl2
            logjacvals[i] = sum(log.(abs.(map( x-> dg(x, nw.nodes[i]), z))))
        end
        return gλvals, logjacvals, dgλvals
    end
    gλvals, logjacvals, dgλvals = jac_comp(btg)
    # @timeit to "gλvals, logjacvals, dgλvals" gλvals, logjacvals, dgλvals = jac_comp(btg)
    # @timeit to "build λbuffer_dict" begin
    λbuffer_dict = Dict{Real, λbuffer}()
    for i = 1:size(gλvals, 2)
        cur_node = nw.nodes[i]
        @assert typeof(cur_node) <: Real #lambda is always real
        push!(λbuffer_dict, cur_node => λbuffer(cur_node, gλvals[:, i], logjacvals[i], dgλvals[:, i]))
    end
    # end
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
        #@info "t1" t1
        if length(t1)==1
            t1 = t1[1]
        end
        cur_train_buf = train_buffer_dict[t1] #get train_buffer
        cur_λ_buf = λbuffer_dict[t2] #get lambda buffer
        θλpair = (t1, t2)::Union{Tuple{Array{T, 1}, T}, Tuple{T, T}} where T<:Real #key used for this buffer
        (_, Σθ_inv_X, qr_Σθ_inv_X, choleskyΣθ, choleskyXΣX) = unpack(cur_train_buf)
        L_inv_X = cur_train_buf.L_inv_X
        (λ, gλz, logjacval) = unpack(cur_λ_buf)
        choleskyXΣX = cur_train_buf.choleskyXΣX
        choleskyΣθ = cur_train_buf.choleskyΣθ
        Fx = getCovariates(train)
        @timeit to "Σθ_inv_y" Σθ_inv_y = choleskyΣθ\gλz
        #Σθ_inv_y = (choleskyΣθ \ gλz) #O(n^2)
        @timeit to "βhat" βhat = choleskyXΣX\(Fx'*Σθ_inv_y)  #O
        #qtilde = (expr = gλz-Fx*βhat; expr'*(choleskyΣθ\expr))
        @timeit to "qtilde" qtilde =  gλz'*Σθ_inv_y  - 2*gλz'*Σθ_inv_X*βhat + βhat'*Fx'*Σθ_inv_X*βhat #O(np) checks out b/c qtilde = norm(remainder)^2
        #remainder = L_inv_y - L_inv_X*βhat
        Σθ_inv_remainder = Σθ_inv_y - Σθ_inv_X*βhat
        cur_θλbuffer = θλbuffer(t1, t2, βhat, qtilde, nothing, Σθ_inv_y, nothing, Σθ_inv_remainder)
        push!(θλbuffer_dict, θλpair => cur_θλbuffer)
    end
        return θλbuffer_dict
end

function init_empty_buffer_dict(nw::nodesWeights, train_buffer_dict::Union{Dict{Array{T, 1}, train_buffer}, Dict{T, train_buffer}} where T<: Real, 
    buf_type::Union{Type{jac_buffer}, Type{test_buffer}})
    d = getDimension(nw)
    if d==1 #if single length scale, then key will be Real, else array
        return Dict{Real, buf_type}(arr => buf_type() for arr in keys(train_buffer_dict))
    else
        return Dict{Array{T, 1} where T<:Real, buf_type}(arr => buf_type() for arr in keys(train_buffer_dict))
    end
end

# """
# Initializes test_buffer_dict with empty test buffers, so that the keys match those of train_buffer
# """
# function init_test_buffer_dict(nw::nodesWeights, train_buffer_dict::Union{Dict{Array{T, 1}, train_buffer}, Dict{T, train_buffer}} where T<: Real)
#     d = getDimension(nw)
#     if d==1 #if single length scale, then key will be Real, else array
#         return Dict{Real, test_buffer}(arr => test_buffer() for arr in keys(train_buffer_dict))
#     else
#         return Dict{Array{T, 1} where T<:Real, test_buffer}(arr => test_buffer() for arr in keys(train_buffer_dict))
#     end
# end

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

"""
Notice that this function is initialized differently from the other validation buffers. It is initialized the same way as 
test_buffer. This is because we can't tell what the contents of the buffers will be when the btg object is created, or rather
when we see that we want to do LOOCV. While other buffers are only initialized once, this buffer will be continually updated
with each new data point. Hence we preallocate all the space and then use update! to change the contents of this buffer. 
Essentially we are initializing the keys here so we don't have to worry about it in update! If we didn't initialize the keys, 
we would have to detect whether they have been initialized or not with each call to update!, or add a field which flags whether
the keys have been initialized. 
"""
function init_validation_test_buffer_dict(nw::nodesWeights, test_buffer_dict::Union{Dict{Array{T, 1} where T<: Real, test_buffer}, Dict{T where T<: Real, test_buffer}})
    d = getDimension(nw)
    if d==1
        return Dict{T where T<: Real, validation_test_buffer}(key => validation_test_buffer() for key in keys(test_buffer_dict)) 
    else
        return Dict{Array{T, 1} where T<: Real, validation_test_buffer}(key => validation_test_buffer() for key in keys(test_buffer_dict)) 
    end
end

function init_validation_λ_buffer_dict()
    return Dict{T where T<: Real, validation_λ_buffer}()
end

######
###### Update train_buffer when extending kernel system (for Bayesian optimization)
###### Update test_buffer for each new problem, parametrized by triple (x0, Fx0, y0)
######



    #Σθ::Array{Float64, 2}  #only top n by n block is filled
    #Σθ_inv_X::Array{Float64, 2}
    #qr_Σθ_inv_X::LinearAlgebra.QRCompactWY{Float64,Array{Float64,2}}
    #choleskyΣθ::IncrementalCholesky
    #choleskyXΣX::Cholesky{Float64,Array{Float64, 2}}
    #logdetΣθ::Float64
    #logdetXΣX::Float64
    #capacity::Int64 #size of Σθ, maximum value of n
    #n::Int64 #number data points incorporated
    #k::AbstractCorrelation
    #θ::Union{Array{T, 1}, T} where T<:Real
    #L_inv_X::Array{Float64, 2}

"""
Updates training buffer with new training data
- 
"""
function update!(train_buffer::train_buffer, trainingData::AbstractTrainingData)  #use incremental cholesky to update training train_buffer
    
    @assert train_buffer.n < trainingData.n #train_train_buffer must be "older" than trainingData
    k = trainingData.n - train_buffer.n  #typically going to be 1
    @assert k == 1 #we only work with single-point updates for now 
    A12, A2 = extend(train_buffer.choleskyΣθ, k) #two view objects 

    #peel off "new" values from already updated trainingData
    x_new = getPosition(trainingData)[end:end, :]
    Fx_new = getCovariates(trainingData)[end:end, :]
    y_new = getLabel(trainingData)[end]

    Bθ_new = cross_correlation(Gaussian(), train_buffer.θ, x_new, getPosition(trainingData)[1:end-1, :])
    Eθ_new = [1]

    #A12 = cross_correlation(train_buffer.k(), train_buffer.θ, trainingData.x[1:end-k], trainingData.x[end-k+1:end])
    #A2 = correlation(train_buffer.k(), train_buffer.θ, trainingData.x[end-k+1:end]) #Σθ should get updated automatically, but only upper triangular portion
    A12 .= Bθ_new' #mutates choleskyΣθ
    A2 .= Eθ_new #mutates choleskyΣθ
    update!(train_buffer.choleskyΣθ, k) #extends Cholesky decomposition -- this is where the action happens
    
    train_Fx = getCovariates(trainingData)
    train_buffer.Σθ_inv_X = train_buffer.choleskyΣθ\train_Fx #potential repeated computations
    train_buffer.choleskyXΣX = cholesky(Hermitian(train_Fx' * train_buffer.Σθ_inv_X))
    train_buffer.logdetΣθ = logdet(train_buffer.choleskyΣθ)
    train_buffer.logdetXΣX = logdet(train_buffer.choleskyXΣX)
    train_buffer.n += k #number of incorporated points
    @assert train_buffer.n == trainingData.n #invariant 
    return nothing 
end

# function update!(train_buffer::train_buffer, test_buffer::test_buffer, trainingData::AbstractTrainingData)  #use incremental cholesky to update training train_buffer
#     @assert typeof(x0)<:Array{T, 2} where T<:Real
#     @assert typeof(Fx0)<:Array{T, 2} where T<:Real
#     @assert typeof(y0)<:Array{T, 1} where T<:Real 
#     @assert train_buffer.n < trainingData.n #train_train_buffer must be "older" than trainingData
#     k = trainingData.n - train_buffer.n  #typically going to be 1
#     @assert k == 1 #we only work with single-point updates for now 
#     A12, A2 = extend(train_buffer.choleskyΣθ, k) #two view objects 
#     #A12 = cross_correlation(train_buffer.k(), train_buffer.θ, trainingData.x[1:end-k], trainingData.x[end-k+1:end])
#     #A2 = correlation(train_buffer.k(), train_buffer.θ, trainingData.x[end-k+1:end]) #Σθ should get updated automatically, but only upper triangular portion
#     A12 .= test_buffer.Bθ' #mutates choleskyΣθ
#     A2 .= test_buffer.Eθ #mutates choleskyΣθ
#     update!(train_buffer.choleskyΣθ, k) #extends Cholesky decomposition -- this is where the action happens
#     train_buffer.Σθ_inv_X = train_buffer.choleskyΣθ\trainingData.Fx
#     train_buffer.choleskyXΣX = cholesky(trainingData.Fx' * train_buffer.Σθ_inv_X)
#     train_buffer.logdetΣθ = logdet(train_buffer.choleskyΣθ)
#     train_buffer.logdetXΣX = logdet(train_buffer.choleskyXΣX)
#     train_buffer.n .+= k #number of incorporated points
#     return nothing 
# end

function refresh_buffer(buffer::Union{test_buffer, jac_buffer})
    buffer.update_bit = true
    return nothing
end

function refresh_buffer_dict(buffer_dict::Union{Union{Dict{Array{T, 1} where T<: Real, test_buffer}, Dict{T where T<: Real, test_buffer}}, 
    Union{Dict{Array{T, 1} where T<: Real, jac_buffer}, Dict{T where T<: Real, jac_buffer}}})
    for key in keys(buffer_dict)
        refresh_buffer(buffer_dict[key])
    end
    return nothing
end

"""
Update test_buffer, which depends on testing data, training data, and train_buffer. 
Note that a test_buffer, attached to a unique theta-value, is only capable of being updated once before being having to be
refreshed again. The logic in this function, which checks init_bit and update_bit, ensure that this invariant holds.
"""
function update!(train_buffer::train_buffer, test_buffer::test_buffer, trainingData::AbstractTrainingData, testingData::AbstractTestingData)
    try 
        @assert checkCompatibility(trainingData, testingData) #make sure testingData is compatible with trainingData
    catch e
        @info "train", trainingData
        @info "test", testingData
    end
    function update_me()
        @timeit to "Eθ" test_buffer.Eθ = correlation(train_buffer.k, train_buffer.θ, testingData.x0)  
        @timeit to "Bθ" Bθ = cross_correlation(train_buffer.k, train_buffer.θ, testingData.x0, getPosition(trainingData))  
        @assert size(Bθ, 2)>= size(Bθ, 1) #row vector, as in the paper
        test_buffer.Bθ = Bθ
        #@info "test_buffer.Bθ'", test_buffer.Bθ'
        @timeit to "ΣθinvBθ" test_buffer.ΣθinvBθ = train_buffer.choleskyΣθ\test_buffer.Bθ'
        @timeit to "Dθ" test_buffer.Dθ = test_buffer.Eθ - test_buffer.Bθ*test_buffer.ΣθinvBθ
        #println("shape of Dtheta: ", size(test_buffer.Dθ ))
        #println("shape of Btheta: ", size(test_buffer.Bθ ))
        #println("shape of Fx0: " , size(testingData.Fx0))
        #println("shape of train_buffer.Σθ_inv_X", size(train_buffer.Σθ_inv_X))
        @timeit to "Hθ" test_buffer.Hθ = testingData.Fx0 - test_buffer.Bθ*(train_buffer.Σθ_inv_X) 
        @timeit to "Cθ" test_buffer.Cθ = test_buffer.Dθ + test_buffer.Hθ*(train_buffer.choleskyXΣX\test_buffer.Hθ') 
        test_buffer.θ = train_buffer.θ
        test_buffer.update_bit = false
        test_buffer.init_bit = true
        #@info "D" test_buffer.Dθ
        #@info "H" test_buffer.Hθ
    end
    if test_buffer.init_bit
        if test_buffer.update_bit
            update_me()     
        end
    else
        update_me()
    end
    return nothing
end

"""
Update jac_buffer
"""
function update!(jac_buffer::jac_buffer, test_data::testingData, train_data::AbstractTrainingData, test_buffer::test_buffer, train_buffer::train_buffer)
    function update_me()
        x0 = test_data.x0
        θ = test_buffer.θ
        Bθ = test_buffer.Bθ
        Hθ = test_buffer.Hθ
        Σθ_inv_X = train_buffer.Σθ_inv_X
        choleskyXΣX = train_buffer.choleskyXΣX
        ΣθinvBθ = test_buffer.ΣθinvBθ

        x = getPosition(train_data)
        n = getNumPts(train_data)

        @timeit to "derivative intermediates" begin
            #@assert typeof(θ) <: Array{T, 1} where T 
            #@assert length(θ) ==d

            S = repeat(x0, n, 1) .- x # n x d 
            d = size(x0, 2)
            #println("size of S: ", size(S))
            flattenedθ = typeof(θ)<:Real ? [θ for i=1:d] : θ[:]
            @timeit to "jacB" jacB =   diagm(Bθ[:]) * S * diagm(- flattenedθ) #n x d
            #println("size of jacB: ", size(jacB))
            #println("size of Bθ: ", size(Bθ))
            #println("size of choleskyΣθ: ", size(choleskyΣθ))
            #println("choleskyΣθ inv Bθ': ", size(choleskyΣθ \ Bθ'))
            #jacD = -2* jacB' * (choleskyΣθ \ Bθ') #d x 1
            @timeit to "jacD" jacD = -2* jacB' * ΣθinvBθ
            #println("size of jacD: ", size(jacD))
            #assuming linear polynomial basis
            @timeit to "jacFx0" jacFx0 = vcat(zeros(1, d), diagm(ones(d))) #p x d
            #println("size of Fx0: ", size(Fx0))
            @timeit to "jacH" jacH = jacFx0' - jacB' * Σθ_inv_X #d x p
            #println("size of jacH: ", size(jacH))
            @timeit to "jacC" jacC = jacD + 2 * jacH * (choleskyXΣX \ Hθ') #d x 1
            jac_buffer.jacB = jacB
            jac_buffer.jacD = jacD
            jac_buffer.jacH = jacH
            jac_buffer.jacC = jacC
        end
        jac_buffer.update_bit = false
        jac_buffer.init_bit = true
    end
    if jac_buffer.init_bit
        if jac_buffer.update_bit
            update_me()     
        end
    else
        update_me()
    end
    return nothing
end


# """
# update validation_test_buffer
# """
# function update!(validation_test_buffer::validation_test_buffer, train_buf::train_buffer, test_buf::test_buffer, validate)
#     ΣθinvBθ = test_buf.ΣθinvBθ
#     choleskyΣθ = train_buf.choleskyΣθ 
#     θ = train_buf.θ
#        # now these are giving me issues vvv
#     (_, _, ΣθinvBθ, _, _, _) = unpack(test_buf) #a critical assumption is that the covariates Fx0 remain constant throughout cross-validation
#        (_, _, _, choleskyΣθ, _, _, _, θ) = unpack(train_buf)
#     ΣθinvBθ = lin_sys_loocv_IC(ΣθinvBθ, choleskyΣθ, validate) #new ΣθinvBθ of dimension n-1 x 1
#     validation_test_buffer.θ = θ
#     validation_test_buffer.ΣθinvBθ = ΣθinvBθ
#     return nothing
# end


