abstract type AbstractTrainingData end
abstract type AbstractTestingData end

"""
Represents training data for GP regression problem. 
"""
struct trainingData{T<:Array{Float64, 2}, S<:Array{Float64, 1}} <: AbstractTrainingData
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
mutable struct extensible_trainingData{T<:Array{Float64, 2}, S<:Array{Float64, 1}} <:AbstractTrainingData
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
mutable struct testingData{T<:Array{Float64, 2}} <:AbstractTestingData
    x0::T
    Fx0::T
end

"""
Initializes and returns extensible container for training data
    * capacity: capacity of extensible container
    * dimension: dimension of data points in container
    * p: dimension of covariate vectors associated with each data point
Used for Bayesian optimization
"""
function newExtensibleTrainingData(dimension::Int64, p::Int64, capacity=100)::extensible_trainingData
    x = Array{Float64}(undef, capacity, dimension)
    Fx = Array{Float64}(undef, capacity, p)
    y = Array{Float64}(undef, capacity)
    n = 0 #number incorporated points
    extensible_trainingData(x, Fx, y, dimension, p, n, capacity)
end

"""
"""
function newTrainingData(x, Fx, y)::trainingData 
    #we will not try to extend the kernel system defined using these quantities, 
    #therefore there is no need to preallocate space
    trainingData(x, Fx, y, size(x, 2),  size(Fx, 2), size(x, 1))
end


"""
Updates extensible training dataset with new data points (locations, covariates, labels)
"""
function update!(e::extensible_trainingData, x0, Fx0, y0)
    @assert typeof(x0)<:Array{T, 2} where T<:Real
    @assert typeof(Fx0)<:Array{T, 2} where T<:Real
    @assert typeof(y0)<:Array{T, 1} where T<:Real
    k = size(x0, 1)
    if e.n + size(x0, 1) > e.capacity
        throw(BoundsError)
    end
    e.x[e.n + 1 : e.n + k, :] = x0
    e.Fx[e.n + 1 : e.n + k, :] = Fx0
    e.y[e.n + 1 : e.n + k] = y0
    e.n += k
    return nothing
end

"""
Replaces prediction location and covariates in testingData with new
new locations and covariates 
"""
function update!(e:: testingData, x0, Fx0)
    @assert typeof(x0)<:Array{T, 2} where T<:Real
    @assert typeof(Fx0)<:Array{T, 2} where T<:Real
    e.x = x0
    e.Fx = Fx0
    return nothing
end


