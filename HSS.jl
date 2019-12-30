
using LinearAlgebra

function isleafnode(i, n)
    # return a boolean value indicating whether i is the leaf node of 
    # the tree in postorder traversal with n nodes
    res = [1]
    level = log(2, n+1)
    for k in 1:level-1
        append!(res, res .+ (2^k - 1))
    end
    return i in res
end

function findchild(i, n)
    # find the child of node i
    # if i is leaf node, return None
    if isleafnode(i, n)
        return nothing
    end
    level = log(2, n+1)
    right = i - 1
    node = [n]
    level -= 1
    while !(i in node)
        node = node .- 1
        diff = 2^level - 1
        temp = node .- diff
        append!(node, temp)
        level -= 1
    end
    left = right - (2^level-1)
    return Int(left), right
end

# Construct kernel matrix into HSS representation

# First try a 4*4 version
function HSS1(H)
    len = size(H,1)
    block = 4
    partition = Int(round(len/block))
    
#     X = []
#     Y = []
#     D = []
#     for i in 1:block
#         l = partition * (i-1) + 1
#         r = max(partition * i, len)
#         Di = H[l:r, l:r]
#         Xi = H[l:r, r+1:len]
#         Yi = H[r+1:len, l:r]
#         push!(X, Xi)
#         push!(Y, Yi)
#         push!(D, Di)
#     end
    
#     U = []
#     V = []
    
    # Node 1
    l = 1
    r = partition
    X1 = H[l:r, (r+1):len]
    Q1 = qr(X1)
    U1 = Matrix(Q1.Q) # U1 = V1, X1 = Y1'
    T = Matrix(Q1.R)
    T12 = T[:, 1:partition]
    T14 = T[:, (partition+1):(partition*2)]
    T15 = T[:, (partition*2+1):end]
    
    # Node 2
    l = partition + 1
    r = partition*2
    X2 = hcat(T12', H[l:r, (r+1):len])
    Q2 = qr(X2)
    U2 = Matrix(Q2.Q) # U2 = V2, X2 = Y2'
    T = Matrix(Q1.R)
    B2 = T[:, 1:size(T12', 2)] # B1 = B2'
    T24 = T[:, size(T12', 2)+1:size(T12', 2)+partition]
    T25 = T[:, size(T12', 2)+partition+1:end] 
    
    # Node 3
    X3 = [T14 T15; T24 T25]
    println(size(X3))
    Q3 = qr(X3)
    R = Matrix(Q3.Q)
    R1 = R[1:size(T14,1),1:partition] # R1 = W1
    R2 = R[size(T14,1)+1:end, 1:partition]
    T = Matrix(Q3.R)
    T34_tilde = T[1:partition, 1:size(T14, 2)]
    T35_tilde = T[1:partition,size(T14, 2)+1:end]
    
    # Node 4 and Node 5
    X4 = [T34_tilde H[partition*2+1:partition*3,partition*3+1:end]]
    Q4 = qr(X4)
    U4 = Matrix(Q4.Q) # U4 = V4
    T = Matrix(Q4.R)
    T43_hat = T[:, 1:partition]
    T45 = T[:, partition+1:end]
    
    X5 = [T35_tilde T45']
    Q5 = qr(X5)
    U5 = Matrix(Q5.Q) # U4 = V4
    T = Matrix(Q5.R)
    T53_hat = T[:, 1:partition]
    B5 = T[:, partition+1:end]
    
    # Node 6
    X6 = [T43_hat; T53_hat]
    Q6 = qr(X6)
    R = Matrix(Q6.Q)
    R4 = R[1:partition,:]
    R5 = R[partition+1:end, :]
    B6 = Matrix(Q6.R)
end 

function factorization(U, D, B, R, n)
    """
    n is the number of nodes in the HSS tree
    U, D, B, R are the list of matrices 
    """
    Ds = []
    Us = []
    L = []
    Q = []
    for i in 1:n-1
        if not isleafnode(i, n)
            c1, c2 = findchild(i, n)
            Di = [Ds[c1] Us[c1]*B[c1]*Us[c2]';
                Us[c2]*B[c1]'*Us[c1]' Ds[c2]]
            Ui = [Us[c1]*R[c1] Us[c2]*R[c2]]
        end
        Qi = I # need to modify
        Usi = Qi'*U[i] 
        Dhat = Qi'*D[i]*Qi
        pi = size(Usi, 1)
        mi = size(Dhat, 1)
        D11 = Dhat[1:mi-pi, 1:mi-pi]
        D12 = Dhat[1:mi-pi, mi-pi+1:end]
        D21 = Dhat[pi+1:end, 1:mi-pi]
        D22 = Dhat[pi+1:end, mi-pi+1:end]
        C = cholesky(D11)
        Li = Matrix(C.L)
        Dsi = D22- D21*L'\(L\D12)
        push!(Us, Usi)
        push!(Ds, Dsi)
        push!(L, Li)
        push!(Q, Qi)
    end
    C = cholesky(D[n])
    Ln = Matrix(C.L)
    push!(L, Ln)
    return L, Q
end 

A = rand(12,12)
A = A'*A
HSS1(A)
