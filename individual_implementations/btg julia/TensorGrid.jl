include("quadrature.jl")
include("Derivatives.jl")
include("TDist.jl")
include("transforms.jl")

"""
    createTensorGrid(example, meshtheta, meshlambda, type)

Define a function ``f`` from ``R^k`` to ``Mat(n, n)``, such that ``f(z_0)_{ij} = p(z_0|z, θ_i, λ_j)``, 
where ``i`` and ``j`` range over the meshgrids over ``θ`` and ``λ``. Optional arg ``type`` is ""Gaussian""
by default. If ``type`` is "Turan", then use Gauss-Turan quadrature to integrate out ``0`` variable. 
"""
function getTensorGrid(train::trainingData{T1, T2}, test::testingData{T1}, priorθ, priorλ, nodesWeightsθ::nodesWeights, nodesWeightsλ::nodesWeights, transform, quadtype::String) where T1 <:Array{Float64} where T2<:Array{Float64} 
    nodesθ = nodesWeightsθ.nodes
    nodesλ = nodesWeightsλ.nodes
    weightsθ = nodesWeightsθ.weights
    weightsλ = nodesWeightsλ.weights
    X = train.X; X0 = test.X0; z = train.z; n = size(X, 1); p = size(X, 2)
    l1 = length(nodesθ); l2 = length(nodesλ); l3 = size(nodesWeightsθ.weights, 2)
    function func_fixed(θ::Float64)
        return funcθ(θ, train, test, quadtype)
    end
    #precompute buffer for θ-values
    theta_param_list = Array{Union{θ_params{Array{Float64, 2}, Cholesky{Float64,Array{Float64, 2}}}, θ_param_derivs{Array{Float64, 2}, Cholesky{Float64,Array{Float64, 2}}}}}(undef, l1)
    for i=1:l1
        theta_param_list[i] = func_fixed(nodesθ[i])
    end

    #initialize tensor grid of weights
    n1 = length(nodesθ); n2 = length(nodesλ); n3 = size(weightsθ, 2); 
    weightsTensorGrid = zeros(n1, n2, n3) #tensor grid of weights
    for i = 1:n1
        for j = 1:n2
            funcs = posterior_theta(nodesθ[i], nodesλ[j], priorθ, priorλ, train, test, boxCoxObj, theta_param_list[i], quadtype)
            for k = 1:n3
                weightsTensorGrid[i, j, k] = weightsθ[i, k]*weightsλ[j]*funcs[k]
            end
        end
    end
    #compute matrix of Jacobians stably (using shift and scale)
    jacvals = zeros(1, n2)
    df = transform.df
    for i = 1:n2
        jacvals[i] = sum(log.(abs.(map( x-> df(x, nodesλ[i]), z))))
    end
    maxz = maximum(jacvals)
    for i = 1:n2
        jacvals[i] = jacvals[i]-maxz
    end
    jacTensorGrid = repeat((exp.(jacvals)).^(1-p/n), n1, 1, n3)

    weightsTensorGrid = weightsTensorGrid .* jacTensorGrid 
    weightsTensorGrid =  weightsTensorGrid/sum(weightsTensorGrid)

    #tensor grid for PDF p(z0|theta, lambda, z)
    tgridpdf = Array{Any, 3}(undef, l1, l2, l3) 
    #tensor grid for CDF P(z0|theta, lambda, z)
    tgridcdf = Array{Any, 3}(undef, l1, l2, l3) 

    for i = 1:l1
        for j = 1:l2 
            funcs = partial_theta(nodesθ[i], nodesλ[j], train, test, boxCoxObj, theta_param_list[i], quadtype)
            for k = 1:l3
                #tgrid[i, j, k] = funcs[k]       
                tgridpdf[i, j, k] = funcs[1][k]
                tgridcdf[i, j, k] = funcs[2][k]
            end
        end
    end
    function evalTgrid_pdf(z0)
        res = Array{Float64, 3}(undef, l1, l2, l3)
        for i=1:l1
            for j = 1:l2
                for k =1:l3
                    res[i, j, k] = tgridpdf[i, j, k](z0)
                end
            end
        end 
        return res
    end
    function evalTgrid_cdf(z0)
        res = Array{Float64, 3}(undef, l1, l2, l3)
        for i=1:l1
            for j = 1:l2
                for k =1:l3
                    res[i, j, k] = tgridcdf[i, j, k](z0)
                end
            end
        end 
        return res
    end
    #product rule loop for Turan quadrature (k>1). In the k=1 case, this reduces to taking the dot
    # product between the tensor grid and weights grid.
    function pdf(z0)
        grid = evalTgrid_pdf(z0)
        res = 0
        for i = 1:n3
            for j = 1:i
                res = binomial(i, j) * dot(grid[:, :, j],  weightsTensorGrid[:, :, n3-j+1])  
            end
        end
        return res
    end
    function cdf(z0)
        grid = evalTgrid_cdf(z0)
        res = 0
        for i = 1:n3
            for j = 1:i
                res = binomial(i, j) * dot(grid[:, :, j], weightsTensorGrid[:, :, n3-j+1])  
            end
        end
        return res
    end
    return (pdf, cdf)
    #return z0 -> dot(evalTgrid_pdf(z0), weightsTensorGrid), z0 -> dot(evalTgrid_cdf(z0), weightsTensorGrid)
end


