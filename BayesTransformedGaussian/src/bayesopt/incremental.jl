using LinearAlgebra

import Base: size, inv, getindex, setindex
import LinearAlgebra: ldiv!, det, logdet, isposdef

#####
##### Incremental Cholesky Decomposition
#####

@doc raw"""
"""
mutable struct IncrementalCholesky{T} <: Factorization{T}
    capacity::Int
    n::Int
    info::Int
    R::Matrix{T}
end
function incremental_cholesky(capacity, c::Number; check=true)
    info = 0
    if c <= 0
        check && throw(LinearAlgebra.PosDefException(1))
        info = -1
    end
    R = Array{typeof(sqrt(c))}(undef, capacity, capacity)
    R[1, 1] = sqrt(c)
    return IncrementalCholesky(capacity, 1, info, R)
end
function incremental_cholesky(capacity, A::AbstractMatrix; check=true)
    n = size(A, 1)
    R = Array{eltype(A)}(undef, capacity, capacity)
    R[1:n, 1:n] .= A
    chol = cholesky!(Symmetric(@view R[1:n, 1:n]))
    return IncrementalCholesky(capacity, n, chol.info, R)
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

function remove_col!(R::IncrementalCholesky)
    @assert R.n - 1 >= 1
    R.n -= 1
    return nothing
end

get_chol(R::IncrementalCholesky) = Cholesky(view(R.R, 1:R.n, 1:R.n), :U, R.info) 

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
    capacity::Int
    n::Int
    A::Array{T,N}
end
function data_array(T, capacity, dims::Vararg{Int, N}) where N
    n = length(dims)
    A = Array{T, N+1}(undef, dims..., capacity)
    return DataArray(capacity, 0, A)
end
function data_array(capacity, A::Array{T, N}) where T where N
    s = size(A)
    new_A = Array{T, N}(undef, ntuple(i -> s[i], N-1)..., capacity)
    new_A[ntuple(i -> 1:s[i], N)...] .= A
    return DataArray(capacity, s[N], new_A)
end

function array_view(A::DataArray{T, N}) where T where N
    return view(A.A, ntuple(_ -> Colon(), N-1)..., 1:A.n)
end


size(A::DataArray, args...) = size(array_view(A), args...)

getindex(A::DataArray, args...) = getindex(array_view(A), args...)

setindex!(A::DataArray, args...) = setindex!(array_view(A), args...)

function extend!(A::DataArray{T, N}, m) where T where N
    @assert A.n + m <= A>capacity
    vw = view(A.A, :, A.n:A.n+m)
    A.n += m
    return vw
end

function add_point!(A::DataArray{T, N}, v) where T where N
    @assert A.n + 1 <= A.capacity
    A.A[ntuple(_ -> Colon(), N-1)..., A.n + 1] .= v
    A.n += 1
    return nothing
end

function add_points!(A::DataArray{T, N}, B) where T where N
    m = size(B)[end]
    @assert A.n + m <= A.capacity
    A.A[ntuple(_ -> Colon(), N-1)..., A.n:A.n+m] .= B
    A.n += m
    return nothing
end

function remove_point!(A::DataArray)
    @assert A.n - 1 >= 1
    A.n -= 1
    return nothing
end

const DataMatrix{T} = DataArray{T, 2}
const DataVector{T} = DataArray{T, 1}
