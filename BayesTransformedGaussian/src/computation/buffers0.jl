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
    capacity::int64 #size of Σθ, maximum value of n
    n::int64
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
"""
function init_train_buffer(θ::Float64, train::trainingData{A, B}, k::AbstractCorrelation, capacity=100) where A<:Array{Float64, 2} where B<:Array{Float64, 1}
    #unpack values from training buffer
    x = train.x
    Fx = train.Fx
    y = train.y
    num = train.num
    p = train.p
    dimension = train.dimension
    
    Σθ = Array{Float64}(undef, capacity, capacity)
    Σθ[1:num, 1:num] = correlation(k(), θ, x) #note that length scale θ is applied on the numerator
    choleskyΣθ = incremental_cholesky!(Σθ, num)
    Σθ_inv_X = (choleskyΣθ\Fx)
    choleskyXΣX = cholesky(Hermitian(X'*(Σθ_inv_X))) #regular cholesky because we don't need to extend this factorization
    return train_buffer(Σθ, Σθ_inv_X, choleskyΣθ, choleskyXΣX, capacity, num)
end

function init_test_buffer(θ::Float64, train::trainingData{A, B}, test:testingData{A}, k::AbstractCorrelation) where A<:Array{Float64, 2} where B<:Array{Float64, 1}
    Eθ = correlation(k(), θ, test.x0)    
    Bθ = cross_correlation(k(), θ, test.x0, train.x)  
    ΣθinvBθ = train.choleskyΣθ\Bθ'
    Dθ = Eθ - Bθ*ΣθinvBθ
    Hθ = X0 - Bθ*(train.Σθ_inv_X) 
    Cθ = Dθ + Hθ*(choleskyXΣX\Hθ') 
    train_buffer(Eθ, Bθ,  ΣθinvBθ, Dθ, Hθ , Cθ)
end

function update_train_buffer!(buffer::train_buffer, x0, Fx0, y0) #use incremental cholesky to update training buffer
    push!(buffer.y, y0)
end

function update_test_buffer!()
end