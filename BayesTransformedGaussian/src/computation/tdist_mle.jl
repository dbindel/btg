include("../btg.jl")
"""
Precompute some quantities
diffs - array of differences between x_i and x_j, used to compute partial_Σθ_theta for various theta
"""
function globalvars(btg::btg)
    n = getNumPts(btg.trainingData)
    d = getDimension(btg.trainingData)
    x = getPosition(btg.trainingData)
    diffs = Array{Array{T, 2} where T, 2}(undef, n, n)
    for i = 1:n
        for j = 1:n
            diffs[i, j] = x[i:i, :] - x[j:j, :]
        end
    end
    return diffs 
end

"""
Takes dot product of vectir a with each entry in matrix of vectors b and stores
result in corresponding entry in b
"""
function entrywise_dot(a, b)
    #@assert size(a, 2) == size(b[1, 1], 2)
    #@assert size(a, 1) == size(b[1, 1], 1)
    a = reshape(a, size(b[1, 1], 1), size(b[1, 1], 2))
    c = copy(b)
    for i = 1:size(c, 1)
        for j = 1:size(c, 2)
            c[i,j] = a .* c[i, j]
        end
    end
    return c
end

function hadamard(a, b)
    @assert size(a, 1) == size(b, 1)
    @assert size(a, 2) == size(b, 2)
    m = size(a, 1)
    n = size(a, 2)
    c = similar(a)
    for i =1:m
        for j = 1:n
            c[i, j] = a[i, j] .* b[i, j]
        end
    end
    return
end

"""
Squares each entry of vector v
"""
function vector_square(v)
    w = copy(v)
    for i = 1:length(v)
        w[i] = v[i]^2
    end
    return w
end

"""
log(p(theta, lambda|z))
"""
function tdist_mle(btg::btg, theta, lambda)
    (x, Fx, y, _, n, p) = unpack(btg.trainingData) #unpack training data
    function Σθ(theta)
        return correlation(btg.k, theta, getPosition(btg.trainingData); jitter = 1e-10)
    end
    g = btg.g 
    dg = (y, λ) -> partialx(btg.g, y, λ)
    choleskyΣθ = cholesky(Σθ(theta))
    gλz = g(y, lambda)
    Σθ_inv_y = choleskyΣθ\gλz
    Σθ_inv_X = choleskyΣθ\Fx
    choleskyXΣX = cholesky(Hermitian(Fx'*Σθ_inv_X))
    pθ = btg.priorθ
    pλ = btg.priorλ
    function βhat()
        return choleskyXΣX\(Fx'*Σθ_inv_y) 
    end
    βhat = βhat()
    function qtilde()
        qtilde =  gλz'*Σθ_inv_y  - 2*gλz'*Σθ_inv_X*βhat + βhat'*Fx'*Σθ_inv_X*βhat
    end
    qtilde = qtilde()
    logdetΣθ = logdet(choleskyΣθ)
    logdetXΣX = logdet(choleskyXΣX)

    logJ = z -> abs(reduce(+, map(x -> log(dg(x, lambda)), z)))
    logJ_ret = logJ(y)

    logpθ = logProb(pθ, theta)
    logpλ = logProb(pλ, lambda)

    logprob = logpθ + logpλ + (1-p/n) * logJ_ret - (n-p)/2 * log(qtilde) - 0.5 * logdetXΣX -0.5 * logdetΣθ
    @info "logpθ", logpθ
    @info "logpλ", logpλ
    @info "(1-p/n) * logJ_ret", (1-p/n) * logJ_ret
    @info "- (n-p)/2 * log(qtilde)", - (n-p)/2 * log(qtilde)
    @info "- 0.5 * logdetXΣX", - 0.5 * logdetXΣX
    @info "-0.5 * logdetΣθ", -0.5 * logdetΣθ 
    @info "logprob", logprob
    return -logprob
end

"""
Maximizes p(theta, lambda|z)
"""
function partial_tdist_mle(btg::btg, theta, lambda)
    ##############################
    ######### derivatives #########
    ##############################
    function partial_Σθ_theta(theta)
        diffs = globalvars(btg);
        square = vector_square.(diffs);
        return -0.5 * map(x ->exp.(x),- 0.5 * sum.(entrywise_dot(theta, square))) .* square;
    end

    function partial_detΣθ_theta()

    end

    function partial_det_XΣX_theta()

    end

    function partial_q_theta()
        g = btg.g 
        dg = (y, λ) -> partialx(btg.g, y, λ)
    end

    function partial_q_lambda()
    end

    function partial_J_lambda()
    end
    
    function partial_p_theta()
        return partialx(btg.priorθ, theta)
    end

    function partial_p_lambda() 
        return partialx(btg.priorλ, lambda)
    end
    
    function compute_betahat_prime_theta(choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, choleskyXΣX::Cholesky{Float64,Array{Float64, 2}}, expr_mid, Σθ_prime, X::Array{Float64,2}, gλz, Σθ_inv_X)
        AA = choleskyXΣX\(expr_mid)*(choleskyXΣX\(X'*(choleskyΣθ\gλz)))
        BB = - (choleskyXΣX\(X'*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\gλz)))))
        βhat_prime_theta = AA + BB
        βhat_prime_theta = reshape(βhat_prime_theta, size(βhat_prime_theta, 1), size(βhat_prime_theta, 2)) #turn 1D array into 2D array
    end

    function compute_qtilde_prime_theta(gλz, X::Array{Float64,2}, βhat, βhat_prime_theta, choleskyΣθ::Cholesky{Float64,Array{Float64, 2}}, Σθ_prime::Array{Float64,2})
        meanvv = gλz - X*βhat
        rr = X*βhat_prime_theta
        AA = (-rr)' * (choleskyΣθ \ meanvv)
        BB = - meanvv' * (choleskyΣθ \ (Σθ_prime * (choleskyΣθ \ meanvv)))
        CC =  meanvv' * (choleskyΣθ \ (-rr))
        qtilde_prime_theta = AA .+ BB .+ CC
    end

end