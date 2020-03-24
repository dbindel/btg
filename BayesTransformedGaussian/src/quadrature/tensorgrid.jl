include("quadrature.jl")
include("../computation/derivatives.jl")
include("../transforms.jl")

"""
``getTensorGrid(train, test, priorθ, priorλ, nodesWeightsθ, nodesWeightsλ, transform, quadtype)``

* quadtype can either be \"Gaussian\" or \"Turan\" (N.B. this refers to integration of θ variable, λ always integrated out using Gaussian quadrature) 

Works by defining a function ``f`` from ``R^k`` to ``Mat(n, n)``, such that ``f(z_0)_{ij} = p(z_0|z, θ_i, λ_j)``, 
where ``i`` and ``j`` range over the meshgrids over ``θ`` and ``λ``. Takes the dot product of this 
matrix function with a weights matrix, ``w_ij = p(θ_i, λ_j|z)`` to obtain PDF and CDF functions.

```julia    
getTensorGrid(train, test, priorθ, priorλ, nodesWeightsθ, nodesWeightsλ, boxCoxObj, "Gaussian") #uses BoxCox family of nonlinear transforms and Gaussian quadrature 
```

"""
function getTensorGrid(integrand, train::trainingData{T1, T2}, test::testingData{T1}, priorθ, priorλ, nodesWeightsθ::nodesWeights, nodesWeightsλ::nodesWeights, transform, quadtype::String) where T1 <:Array{Float64} where T2<:Array{Float64} 
    nodesθ = nodesWeightsθ.nodes
    nodesλ = nodesWeightsλ.nodes
    weightsθ = nodesWeightsθ.weights
    weightsλ = nodesWeightsλ.weights
    X = train.X; X0 = test.X0; z = train.z; n = size(X, 1); p = size(X, 2)
    l1 = length(nodesθ); l2 = length(nodesλ); l3 = size(nodesWeightsθ.weights, 2)
    #initializes buffer of theta-dependent quantities
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

    if quadtype == "Gaussian"
    #|Σθ|, qtilde, and Jac(z) are prone to overflow and underflow. As such, we work strictly with their exponents, 
    #and shift them so the largest sum-of-exponents (exponentiating then gives 1). After that, we can safely
    #normalize the weight so that they sum to 1. 
        #compute exponents of qtilde^(-(n-p)/2)
        qvals = zeros(n1, n2)
        g = transform.f; dg = transform.df
        gλvals = zeros(length(z), n2)
        lmbda = λ -> g(z, λ)
        for i = 1:n2 
            gλvals[:, i] = lmbda(nodesλ[i])
        end
        for i = 1:n1
            for j = 1:n2
                #compute auxiliary quantities 
                choleskyXΣX = theta_param_list[i].choleskyXΣX
                choleskyΣθ = theta_param_list[i].choleskyΣθ
                gλz = gλvals[:, j]
                βhat = choleskyXΣX\(X'*(choleskyΣθ\gλz)) 
                qtilde = (expr = gλz-X*βhat; expr'*(choleskyΣθ\expr)) 
                qvals[i, j] = -(n-p)/2 * log(qtilde) 
            end
        end
        qTensorGrid = repeat(qvals, 1, 1, n3)

        #compute exponents of |Σθ|^(-1/2)    
        detvals = zeros(n1, 1)
        for i = 1:n1
            choleskyΣθ = theta_param_list[i].choleskyΣθ
            detvals[i] = sum(log.(diag(choleskyΣθ.U)))
        end
        #we multiply by -1.0 instead of -0.5, because the determinant is the 
        #product of the squares of the diagonal entries of choleskyΣθ.U 
        detTensorGrid = repeat((- 1.0 .* detvals), 1, n2, n3)

        #compute exponents of (Jac(z)^(1-p/n)
        jacvals = zeros(1, n2)
        df = transform.df
        for i = 1:n2
            jacvals[i] = sum(log.(abs.(map( x-> df(x, nodesλ[i]), z))))
        end
        jacTensorGrid = repeat((1-p/n) .* jacvals, n1, 1, n3)

        powerGrid =  qTensorGrid + detTensorGrid + jacTensorGrid #sum of exponents
        powerGrid = exp.(powerGrid .- maximum(powerGrid)) #linear scaling
        powerGrid = powerGrid .* weightsTensorGrid
        weightsTensorGrid =  powerGrid/sum(powerGrid) #normalized grid of weights
            
    elseif quadtype== "Turan"
        weightsTensorGrid =  weightsTensorGrid/sum(weightsTensorGrid) #normalized grid of weights
    end

    #tensor grid for PDF p(z0|theta, lambda, z)
    tgridpdf = Array{Any, 3}(undef, l1, l2, l3) 
    #tensor grid for CDF P(z0|theta, lambda, z)
    tgridcdf = Array{Any, 3}(undef, l1, l2, l3) 

    for i = 1:l1
        for j = 1:l2 
            funcs = integrand(nodesθ[i], nodesλ[j], train, test, boxCoxObj, theta_param_list[i], quadtype)
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
                    #println("tgridpdf(z0): ", tgridpdf[i, j, k](z0))
                    res[i, j, k] = tgridpdf[i, j, k](z0)[1]
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


