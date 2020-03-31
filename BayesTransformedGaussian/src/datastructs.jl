abstract type AbstractData end

"""
Represents training data for GP regression problem. 
"""
struct trainingData{T<:Array{Float64, 2}, S<:Array{Float64, 1}} <: AbstractData
    x::T #matrix of horizontal location vectors stacked vertically
    Fx::T #matrix of covariates
    y::S #array of labels
    d::Int64 #dimension of location vectors in x
    p::Int64 #dimension of covariate vectors in Fx
    num:: Int64 # number of location vectors in x
end

"""
Represents mutable training dataset which can be extended with each new incorporated observation
- Makes preallocations based on capacity
- Used for for Bayesian optimization  
"""
mutable struct extensible_trainingData{T<:Array{Float64, 2}, S<:Array{Float64, 1}} <:AbstractData
    x::T 
    Fx::T 
    y::S 
    d::Int64 
    p::Int64 
    n::Int64 #number of incorporated points so far
    capacity::Int64 #amount of preallocated space
end

"""
Represents a set of testing data. Currently supports single-point prediction.
"""
mutable struct testingData{T<:Array{Float64, 2}} <:AbstractData
    x0::T
    Fx0::T
end

"""
Initializes and returns extensible container for training data
    * capacity: capacity of extensible container
    * dimension: dimension of data points in container
    * p: dimension of covariate vectors associated with each data point
"""
function newExtensibleTrainingData(capacity::Int64, dimension, p)::trainingData
    x = Array{Float64}(Undef, capacity, dimension)
    Fx = Array{Float64}(Undef, capacity, p)
    y = Array{Float64}(Undef, capacity)
    return extensible_trainingData(x, Fx, y, dimension, p, 0, capacity)
end

function update!(e::extensible_trainingData, x0, Fx0, y0)
    @assert typeof(x0)<:Array{Any, 2}
    @assert typeof(Fx0)<:Array{Any, 2}
    @assert typeof(y0)<:Array{Any, 1}
    k = size(x0, 1)
    if e.n +size(x0, 1) > e.capacity
        throw(Error("Maximum training data capacity reached."))
    end
    e.x[e.n + 1 : e.n + k, :] = x0
    e.Fx[e.n + 1 : e.n + k, :] = Fx0
    e.y[e.n + 1 : e.n + k] = y0
    e.n += k
    return nothing
end



function newTrainingData(x, Fx, y)::trainingData 
    #we will not try to extend the kernel system defidne using these quantities, 
    #therefore there is no need to preallocate space
    trainingData(x, Fx, y, size(x, 2),  size(Fx, 2), size(x, 1))
end

function updateTrainingData!(obj::trainingData, x, Fx, y)
    obj.x = x
    obj.Fx = Fx
    obj.y = y
    obj.dimension = size(x, 2)
    obj.p = size(Fx, 2)
    obj.num = size(x, 1)
end

function newTestingData( x0, Fx0)::testingData{Array{Float64, 2}}
    testingData(x0, Fx0)
end

"""
Setter for TestingData object
"""
function setTestingData!(obj::testingData, x0, Fx0)
    obj.x0 = x0
    obj.Fx0 = Fx0
end

