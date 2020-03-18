"""
TODO:
For multiple length scales, make rangeθ multidimensional
"""
function getWeightsGrid(example::setting{Array{Float64, 2}, Array{Float64, 1}}, nodesWeightsθ::Array{Float64, 2}, nodesWeightsλ::Array{Float64, 2})
    nodesθ = nodesWeightsθ.nodes
    weightsθ = nodesWeightsθ.weights
    nodesλ = nodesWeightsλ.nodes
    weightsλ = nodesWeightsλ.weights

    n1 = length(nodesθ); n2 = length(nodesλ); n3 = size(weightsθ, 2); 
    weightsTensorGrid = zeros(n1, n2, n3) #tensor grid of weights
    for i = 1:n1
        for j = 1:n2
            for k = 1:n3
                weightsTensorGrid[i, j, k] = weightsθ[i, k]*weightsλ[j]
            end
        end
    end
    jacTensorGrid = zeros(n2, 1)
    

    jac = z0 -> (abs(reduce( *, map(x -> dg(x, λ), z0))))
    jacz = jac(z)
    jacz^(1-p/n)

end
