using LinearAlgebra

import Base: size, inv
import LinearAlgebra: ldiv!, det, logdet, isposdef

@doc raw"""
"""
mutable struct IncrementalCholesky{T} <: Factorization{T}
    capacity::Int
    n::Int
    info::Int
    R::Matrix{T}
end
function incremental_cholesky(capacity, c; check=true)
    info = 0
    if c <= 0
        check && throw(LinearAlgebra.PosDefException(1))
        info = -1
    end
    R = Array{typeof(sqrt(c))}(undef, capacity, capacity)
    R[1, 1] = sqrt(c)
    return IncrementalCholesky(capacity, 1, info, R)
end

function add_col!(R::IncrementalCholesky, v; check=true)
    @assert R.n + 1 <= R.capacity
    n = R.n
    @views R.R[1:n, n+1] = R \ v[1:end-1]
    @views R.R[n+1, n+1] = v[end] - dot(R.R[1:n, n+1], R.R[1:n, n+1])
    if R.R[n+1, n+1] <= 0
        check && throw(LinearAlgebra.PosDefException(n+1))
        R.info = -1
    end
    R.n += 1
    return nothing
end

get_chol(R::IncrementalCholesky) = Cholesky(view(R.R, 1:R.n, 1:R.n), :U, R.info) 

size(R::IncrementalCholesky, args...) = size(get_chol(R), args...) 

ldiv!(R::IncrementalCholesky, args...) = ldiv!(get_chol(R), args...)

inv(R::IncrementalCholesky) = inv(get_chol(R))

det(R::IncrementalCholesky) = det(get_chol(R))

logdet(R::IncrementalCholesky) = logdet(get_chol(R))

isposdef(R::IncrementalCholesky) = R.info == 0

mutable struct IncrementalColumns{T} <: AbstractMatrix{T}
    capacity::Int
    n::Int
    A::Matrix{T}
end
function incremental_columns(T, capacity, d)
    A = Array{T}(undef, d, capacity)
    return IncrementalColumns(capacity, 0, A)
end

function add_col!(A::IncrementalColumns, v; check=true)
    @assert A.n + 1 <= A.capacity
    A.A[:, A.n + 1] = v
    A.n += 1
    return nothing
end

get_mat(A::IncrementalColumns) = view(A.A, :, 1:A.n)

size(A::IncrementalColumns, args...) = size(get_mat(A), args...)

getindex(A::IncrementalColumns, args...) = getindex(get_mat(A), args...)

setindex!(A::IncrementalColumns, args...) = setindex!(get_mat(A), args...)
    
