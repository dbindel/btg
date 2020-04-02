using LinearAlgebra
include("../datastructs.jl")
include("../kernels/kernel.jl")
include("../bayesopt/incremental.jl")
include("../quadrature/quadrature.jl")

"""
Buffer of θ-dependent quantities
"""
mutable struct train_buffer
    Σθ::Array{Float64, 2}  #only top n by n block is filled
    Σθ_inv_X::Array{Float64, 2}
    choleskyΣθ::IncrementalCholesky
    choleskyXΣX::Cholesky{Float64,Array{Float64, 2}}
    capacity::Int64 #size of Σθ, maximum value of n
    n::Int64 #number data points incorporated
    k::AbstractCorrelation
    θ::Array{Real, 1}
    function train_buffer(θ::Array{Real, 1}, train::AbstractTrainingData, corr::AbstractCorrelation = Gaussian())
        x = train.x
        Fx = train.Fx
        n = train.n
        capacity = typeof(train)<: extensible_trainingData ? train.capacity : n #if not extensible training type, then set size of buffer to be number of data points
        Σθ = Array{Float64}(undef, capacity, capacity)
        Σθ[1:n, 1:n] = correlation(corr, θ, x[1:n, :]) #note that length scale θ is applied on the numerator
        println(Σθ[1:n, 1:n])
        choleskyΣθ = incremental_cholesky!(Σθ, n)
        Σθ_inv_X = (choleskyΣθ\Fx)
        choleskyXΣX = cholesky(Hermitian(Fx'*(Σθ_inv_X))) #regular cholesky because we don't need to extend this factorization
        new(Σθ, Σθ_inv_X, choleskyΣθ, choleskyXΣX, capacity, n, corr, θ)
    end
end

"""
Buffer of (θ, testingData)-dependent quantities 
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

unpack(b::train_buffer) = (b.Σθ, b.Σθ_inv_X, b.choleskyΣθ, b.choleskyXΣX)
unpack(b::test_buffer) = (b.Eθ, b.Bθ, b.ΣθinvBθ, b.Dθ, b.Hθ, b.Cθ)

"""
Initialize dictionary of training buffers for a collection of θ-values corresponding to all quadrature nodes in a nodesWeights object
"""
function init_train_buffer_dict(nw::nodesWeights, trainingData::AbstractTrainingData, corr::AbstractCorrelation = Gaussian())::Dict{Union{Array{Real, 1}, Real}, T} where T<:train_buffer
    train_buffer_dict = Dict{Union{Array{Real, 1}, Real}, train_buffer}
    nodeSet = Set(nw.nodes[i, j] for i = 1:size(nw, 1) for j = 1:size(nw, 2))
    for node in nodeSet
        push!(train_buffer_dict, node => train_buffer(node, trainingData, corr))
    end
    return train_buffer_dict 
end

"""
Initialize dictionary of training buffers for a collection of θ-values corresponding to all quadrature nodes in a nodesWeights object
"""
function init_test_buffer_dict(nw::nodesWeights, train::AbstractTrainingData, test::AbstractTestingData, corr)
    test_buffer_dict = Dict{Real, test_buffer}
    nodeSet = Set(nw.nodes[i, j] for i = 1:size(nw, 1) for j = 1:size(nw, 2))
    for node in nodeSet
        push!(test_buffer_dict, node => init_test_buffer(node, train, test, corr))
    end
    return test_buffer_dict
end

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
Update second buffer, which depends on testing data, training data, and first buffer. 
"""
function update!(train_buffer::train_buffer, test_buffer::test_buffer, trainingData::AbstractTrainingData, testingData::AbstractTestingData)
    test_buffer.Eθ = correlation(trainingData.k, trainingData.θ, testingData.x0)    
    test_buffer.Bθ = cross_correlation(trainingData.k, train_buffer.θ, testingData.x0, trainingData.x)  
    test_buffer.ΣθinvBθ = train_buffer.choleskyΣθ\Bθ'
    test_buffer.Dθ = Eθ - Bθ*ΣθinvBθ
    test_buffer.Hθ = testingData.Fx0 - Bθ*(train_buffer.Σθ_inv_X) 
    test_buffer.Cθ = Dθ + Hθ*(train_buffer.choleskyXΣX\Hθ') 
    return nothing
end