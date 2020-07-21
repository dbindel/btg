include("datastructs.jl")
include("transforms/transforms.jl")

mutable struct GP
    trainingData::AbstractTrainingData #x, Fx, y, p (dimension of each covariate vector), dimension (dimension of each location vector)
    testingData::AbstractTestingData 
    n::Int64 #number of points in kernel system, if 0 then uninitialized
    p::Int64
    g:: AbstractTransform #transform family, e.g. BoxCox(), Identity()
    k::AbstractCorrelation
    function new(trainingData::AbstractTrainingData, testingData::AbstractTestingData, g:: AbstractTransform, k::AbstractCorrelation)
        
    end
end

function predict_y(GP, x_test, y_test)

end

#gp = GP(train, test, IdentityTransform(), Gaussian())