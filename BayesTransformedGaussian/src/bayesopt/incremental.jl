using LinearAlgebra

import Base: size, inv, getindex, setindex
import LinearAlgebra: ldiv!, det, logdet, isposdef

#####
##### Incremental Cholesky Decomposition
#####

@doc raw"""
"""
mutable struct IncrementalCholesky{T} <: Factorization{T}
    n::Int
    info::Int
    R::Matrix{T}
end


function incremental_cholesky!(A::AbstractMatrix, n; check=true)
    size(A, 1) == size(A, 2) || throw(ErrorException("A is not square"))
    @views chol = cholesky!(Symmetric(A[1:n, 1:n]); check=check)
    return IncrementalCholesky(n, chol.info, A)
end

function incremental_cholesky(T, capacity)
    return incremental_cholesky!(Array{T}(undef, capacity, capacity), 0)
end

valid_update(R::IncrementalCholesky, k) = 0 <= R.n + k <= size(R.R, 1) 

function extend(R::IncrementalCholesky, k)
    valid_update(R, k) || throw(ErrorException("Size cannot exceed capacity"))
    return @views R.R[1:R.n, R.n+1:R.n+k], R.R[R.n+1:R.n+k, R.n+1:R.n+k]
end


function update!(R::IncrementalCholesky, k; check=true)
    valid_update(R, k) || throw(ErrorException("Size cannot exceed capacity"))
    @views A12, A2 = R.R[1:R.n, R.n+1:R.n+k], R.R[R.n+1:R.n+k, R.n+1:R.n+k]
    
    ldiv!(get_chol(R).L, A12)
    mul!(A2, A12', A12, -1, 1)
    
    chol = cholesky!(Symmetric(A2); check=check)
    R.info = chol.info
    R.n += k
    return nothing
end

function remove!(R::IncrementalCholesky, k)
    valid_update(R, -k) || throw(ErrorException("Size cannot be negative"))
    R.n -= k
    return nothing
end

cap(R::IncrementalCholesky) = size(R.R, 1)

#error handling slow in julia
get_chol(R::IncrementalCholesky) = Cholesky(view(R.R, 1:R.n, 1:R.n), :U, R.info) #info indicates if matrix is still positive definite

size(R::IncrementalCholesky, args...) = size(get_chol(R), args...) 

ldiv!(R::IncrementalCholesky, args...) = ldiv!(get_chol(R), args...)

inv(R::IncrementalCholesky) = inv(get_chol(R))

det(R::IncrementalCholesky) = det(get_chol(R))

logdet(R::IncrementalCholesky) = logdet(get_chol(R))

isposdef(R::IncrementalCholesky) = R.info == 0

#####
##### Growable Data Array
##### 

mutable struct DataArray{T,N} <: AbstractArray{T,N}
    n::Int
    A::Array{T,N} 
end

data_array(A::Array, n) = DataArray(n, A)
data_array(T, capacity, dims...) = DataArray(0, Array{T}(undef, dims..., capacity))

function array_view(A::DataArray)
    return view(A.A, ntuple(_ -> Colon(), ndims(A) - 1)..., 1:A.n)
end

cap(A::DataArray) = size(A.A)[end]

size(A::DataArray, args...) = size(array_view(A), args...)

getindex(A::DataArray, args...) = getindex(array_view(A), args...)

setindex!(A::DataArray, args...) = setindex!(array_view(A), args...)

valid_update(A::DataArray, k) = 0 <= A.n + k <= size(A.A)[end] 

function extend(A::DataArray, k)
    valid_update(A, k) || throw(ErrorException("Size cannot exceed capacity"))
    return view(A.A, :, A.n+1:A.n+k)
end

function update!(A::DataArray, k)
    valid_update(A, k) || throw(ErrorException("Size cannot exceed capacity"))
    A.n += k
    return nothing
end

function remove!(A::DataArray, k)
    valid_update(A, -k) || throw(ErrorException("Size cannot be negative"))
    A.n -= k
    return nothing
end

const DataMatrix{T} = DataArray{T, 2}
const DataVector{T} = DataArray{T, 1}
