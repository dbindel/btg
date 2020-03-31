using LinearAlgebra
include("../datastructs.jl")
include("../kernels/kernel.jl")
include("../bayesopt/incremental.jl")

"""
Buffer of θ-dependent parameters
"""
struct train_buffer{O<:Array{Float64, 2}, C<:IncrementalCholesky, D<:Cholesky{Float64,Array{Float64, 2}}}
    Σθ::O  #only top n by n block is filled
    Σθ_inv_X::O
    choleskyΣθ::C
    choleskyXΣX::D
    capacity::Int64 #size of Σθ, maximum value of n
    n::Int64 #number data points incorporated
    k::AbstractCorrelation
    θ::Float64
end

mutable struct test_buffer{O<:Array{Float64, 2}}
    Eθ::O
    Bθ::O
    ΣθinvBθ::O
    Dθ::O
    Hθ::O
    Cθ::O
end

"""
Initializes a train_buffer with prescribed capacity. Should only be called once in the lifetime of the train_buffer object.

If want to update model, first update training data object and then update buffers.
"""
function init_train_buffer(θ::Float64, train::AbstractTrainingData, corr::AbstractCorrelation = Gaussian())::train_buffer
    #unpack values from training buffer
    x = train.x
    Fx = train.Fx
    n = train.n
    capacity = typeof(train)<: extensible_trainingData ? train.capacity : n #if not extensible training type, then set size of buffer to be number of data points
    Σθ = Array{Float64}(undef, capacity, capacity)
    Σθ[1:n, 1:n] = correlation(corr, θ, x[1:n, :]) #note that length scale θ is applied on the nerator
    println(Σθ[1:n, 1:n])
    choleskyΣθ = incremental_cholesky!(Σθ, n)
    Σθ_inv_X = (choleskyΣθ\Fx)
    choleskyXΣX = cholesky(Hermitian(Fx'*(Σθ_inv_X))) #regular cholesky because we don't need to extend this factorization
    return train_buffer(Σθ, Σθ_inv_X, choleskyΣθ, choleskyXΣX, capacity, n, corr, θ)
end

function init_test_buffer(θ::Float64, train::trainingData{A, B}, test::testingData{A}, corr::AbstractCorrelation)::test_buffer where A<:Array{Float64, 2} where B<:Array{Float64, 1}
    Eθ = correlation(corr, θ, test.x0)    
    Bθ = cross_correlation(corr, θ, test.x0, train.x)  
    ΣθinvBθ = train.choleskyΣθ\Bθ'
    Dθ = Eθ - Bθ*ΣθinvBθ
    Hθ = X0 - Bθ*(train.Σθ_inv_X) 
    Cθ = Dθ + Hθ*(choleskyXΣX\Hθ') 
    train_buffer(Eθ, Bθ, ΣθinvBθ, Dθ, Hθ , Cθ)
end

"""
Updates antiquated training buffer with updated trainingData  
"""
function update_train_buffer!(buffer::train_buffer, trainingData::AbstractTrainingData) #use incremental cholesky to update training buffer
    @assert typeof(x0)<:Array{T, 2} where T<:Real
    @assert typeof(Fx0)<:Array{T, 2} where T<:Real
    @assert typeof(y0)<:Array{T, 1} where T<:Real 
    @assert buffer.n < trainingData.n #train_buffer must be "older" than trainingData
    k = trainingData.n - buffer.n 
    A12, A2 = extend!(buffer.choleskyΣθ, k) #two view objects 
    A12 = cross_correlation(buffer.k(), buffer.θ, trainingData.x[1:end-k], trainingData.x[end-k+1:end])
    A2 = correlation(buffer.k(), buffer.θ, trainingData.x[end-k+1:end]) #Σθ should get updated automatically, but only upper triangular portion
    update!(buffer.choleskyΣθ, k)
    buffer.Σθ_inv_X = buffer.choleskyΣθ\trainingData.Fx
    buffer.n += k
    return nothing
end

function update_test_buffer!()
    #TODO
end