include("quadrature.jl")

"""
TODO:
For multiple length scales, make rangeθ multidimensional
"""
function getWeightsGrid(setting::setting{Array{Float64, 2}, Array{Float64, 1}}, priorθ, priorλ, nodesWeightsθ::G, nodesWeightsλ::G, nonlinearFuncs)::Array{Float64} where G <:Union{nodesWeights{Array{Float64, 2}}, nodesWeights{Array{Float64, 1}}}
    #unpack nodesWeights objects and setting object
    nodesθ = nodesWeightsθ.nodes
    weightsθ = nodesWeightsθ.weights
    nodesλ = nodesWeightsλ.nodes
    weightsλ = nodesWeightsλ.weights
    X = setting.X; X0 = setting.X0; z = setting.z; n = size(X, 1); p = size(X, 2)
    #initialize tensor grid of weights
    n1 = length(nodesθ); n2 = length(nodesλ); n3 = size(weightsθ, 2); 
    weightsTensorGrid = zeros(n1, n2, n3) #tensor grid of weights
    for i = 1:n1
        for j = 1:n2
            funcs = posterior_theta(nodesθ[i], nodesλ[j], priorθ, priorλ, setting, theta_params)
            for k = 1:n3
                weightsTensorGrid[i, j, k] = weightsθ[i, k]*weightsλ[j]*funcs[k]
            end
        end
    end
    #compute matrix of Jacobians stably (using shift and scale)
    jacvals = zeros(1, n2)
    df = nonlinearFuncs.df
    for i = 1:n2
        jacvals[i] = sum(log.(abs.(map( x-> df(x, nodesλ[i]), z))))
    end
    maxz = maximum(jacvals)
    for i = 1:n2
        jacvals[i] = jacvals[i]-maxz
    end
    jacTensorGrid = repeat((exp.(jacvals)).^(1-p/n), n1, 1, n3)
    #take hadamard product with tensor grid of Jacobians and normalize 
    println("max jac Tensor grid", maximum(jacTensorGrid)) #sanity check
    weightsTensorGrid = weightsTensorGrid .* jacTensorGrid 
    weightsTensorGrid =  weightsTensorGrid/sum(weightsTensorGrid)
end
