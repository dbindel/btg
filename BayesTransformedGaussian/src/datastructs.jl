abstract type AbstractTrainingData end
abstract type AbstractTestingData end

"""
Represents training dataset for GP regression problem. 
"""
struct trainingData{T<:Array{Float64, 2}, S<:Array{Float64, 1}} <: AbstractTrainingData
    x::T #matrix of horizontal location vectors stacked vertically
    Fx::T #matrix of covariates
    y::S #array of labels
    d::Int64 #dimension of location vectors in x
    p::Int64 #dimension of covariate vectors in Fx
    n:: Int64 # number of incorporated points
    trainingData(x, Fx, y) = new(x, Fx, y, size(x, 2), size(Fx, 2), size(x, 1))
end
getLabel(td::trainingData) = td.y
getPosition(td::trainingData) = td.x
getCovariates(td::trainingData) = td.Fx

"""
Represents mutable training dataset which can be extended with each newly incorporated observation 
First inner constructor initializes and returns empty extensible container for training data
Second inner constructor initializes filled extensible container using x, Fx, y
"""
mutable struct extensible_trainingData<:AbstractTrainingData
    x::Array{Float64, 2} #locations
    Fx::Array{Float64, 2} #covariates
    y::Array{Float64, 1} #labels
    d::Int64 #dimension of data points in container
    p::Int64 #dimension of covariate vectors associated with each data point
    n::Int64 #number of incorporated points so far
    capacity::Int64 #amount of preallocated space
    function extensible_trainingData(d::Int64, p::Int64, capacity=100)::extensible_trainingData
        x = Array{Float64}(undef, capacity, dimension)
        Fx = Array{Float64}(undef, capacity, p)
        y = Array{Float64}(undef, capacity)
        n = 0 #number incorporated points
        new(x, Fx, y, d, p, n, capacity)
    end
    function extensible_trainingData(x, Fx, y, capacity=100)
        new(x, Fx, y, size(x, 2), size(Fx, 2), size(x, 1), capacity)
    end
end

"""
Represents a set of testing data. Currently supports single-point prediction.
"""
mutable struct testingData{T<:Array{Float64, 2}} <:AbstractTestingData
    x0::T
    Fx0::T
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
function update!(e:: AbstracttestingData, x0, Fx0)
    @assert typeof(x0)<:Array{T, 2} where T<:Real
    @assert typeof(Fx0)<:Array{T, 2} where T<:Real
    e.x = x0
    e.Fx = Fx0
    return nothing
end

"""
Get capacity of extensible training object or number of data points in vanilla training object
"""
function getCapacity(e::AbstractTrainingData)
    typeof(e)<:ExtensibleTrainingData ? e.capacity : e.n
end