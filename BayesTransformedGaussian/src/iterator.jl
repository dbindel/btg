"""
Returns an iterator over all combinations of theta-lambda quadrature nodes in the form of a CartesianIndices object
Returns empty weightsTensorGrid array for storing tensor quadrature weights
"""
function get_btg_iterator(nwθ::nodesWeights, nwλ::nodesWeights, quadType::Array{String,1})
    nt1 = nwθ.d   #number of dimensions of theta 
    nt2 = nwθ.num #number of theta quadrature in each dimension
    nl1 = nwλ.d   #number of dimensions of lambda 
    nl2 = nwλ.num #number of lambda quadrature in each dimension
    if endswith(quadType[1], "MonteCarlo") && endswith(quadType[2], "MonteCarlo")
        weightsTensorGrid = Array{Float64, 1}(undef, nt2) 
        R = CartesianIndices(weightsTensorGrid)
    elseif quadType == ["Gaussian", "Gaussian"]
        weightsTensorGrid = Array{Float64, nt1+nl1}(undef, Tuple(vcat([nt2 for i = 1:nt1], [nl2 for i = 1:nl1]))) #initialize tensor grid
        R = CartesianIndices(weightsTensorGrid)
        for I in R #I is multi-index
            weightsTensorGrid[I] = getProd(nwθ.weights, nwλ.weights, I) #this step can be simplified because the tensor is symmetric (weights are the same along each dimension)
        end
    else
        weightsTensorGrid = Array{Float64, 2}(undef, nt2, nl2) 
        R = CartesianIndices(weightsTensorGrid)
        weightsTensorGrid = repeat(nwλ.weights, nt2, 1) # add lambda weights 
    end
    return R, weightsTensorGrid #R is a CartesianIndices iterator over weightsTensorGrid
end

"""
Takes in I and interprets it in the context of quadtype pair and (θ, λ)-nodeWeights objects stored in btg.
Returns (θ, λ) index slices (r1 and r2) and (θ, λ) nodes (t1 and t2)
"""
function get_index_slices(nwθ::nodesWeights, nwλ::nodesWeights, quadType::Array{String,1}, I)
    r1 = (endswith(quadType[1], "MonteCarlo") && endswith(quadType[2], "MonteCarlo")) ? I : Tuple(I)[1:end-1] #first n-1 coords or everything
    r2 = Tuple(I)[end] #last coord
    t1 = quadType[1] == "Gaussian" ? getNodeSequence(getNodes(nwθ), r1) : (temp = getNodes(nwθ)[:, r1[1]]; length(temp)==1 ? temp[1] : temp ) #theta node combo
    t2 = getNodeSequence(getNodes(nwλ), r2)
    #t2 = getNodes(nwλ)[:, r2] #lambda node
    if length(t1)==1
        @assert typeof(t1)<:Real
    end
    @assert typeof(t2)<:Real
    return (r1, r2, t1, t2) #The pair (t1, t2) is used a key for various dictionaries
end

