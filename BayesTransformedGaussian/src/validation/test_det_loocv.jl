include("det_loocv.jl")
using Test

#####
##### test det_loocv
#####
#the inremental_cholesky is just a wrapper type that allows one to easily extend the cholesky
#decomposition without allocation more space at each iteration, because a fixed amount of space 
#is allocated at the beginning, and what you get to use is a view of the top left n x n corner
#println("test det_loocv.jl...")
n=5
A = rand(n, n); A = A'*A + UniformScaling(4.0)
#IC = incremental_cholesky!(A, n-1) #make it n-1 just for kicks
IC = cholesky(A)

vals = zeros(1, n)
vals1 = zeros(1, n)
for i=1:n
    subdet = det(A[[1:i-1;i+1:end], [1:i-1;i+1:end]]) 
    vals[i] = subdet
    #println("actual subdets $i: ", subdet)
end

det_original = det(IC)
for i = 1:n
    subdet = det_loocv(IC, det_original, i)
    vals1[i] = subdet
    #println("fast subdets $i: ", subdet)
end

@testset "det_loocv.jl" begin
    @test norm(vals-vals1)<1e-10
end

#####
##### test det_XΣX_loocv
#####
#println("test det_XΣX_loocv...")

n = 20
m = 3
X = rand(n, m)
A = (w = rand(n, n);w'*w + UniformScaling(5))
cholA = cholesky(A)
detA = det(cholA)
bigmat = [A X; X' zeros(m, m)]
#for i = 1:n
#    println("actual subdet $i: ", det(bigmat[[1:i-1;i+1:end], [1:i-1;i+1:end]]))
#end
vals = zeros(1, n)
vals1 = zeros(1, n)
for i = 1:n
    X_minus_i = X[[1:i-1;i+1:end], :]
    sigma_minus_i = A[[1:i-1;i+1:end], [1:i-1;i+1:end]]
    vals[i] = det(X_minus_i'*(sigma_minus_i\X_minus_i))
    #println("actual subdet $i: ", det(X_minus_i'*(sigma_minus_i\X_minus_i)))
end
for i =1:n
    subdet =  det_XΣX_loocv(X, cholA, detA, i)
    vals1[i] = subdet
    #println("fast subdet $i: ", subdet)
end

@testset "det_XΣX_loocv" begin
    @test norm(vals-vals1)<1e-10
end

#####
##### test logdet_loocv_IC
#####
#println("test logdet_loocv_IC...")
n=5
A = rand(n, n); A = A'*A + UniformScaling(4.0)
B = copy(A)
IC = incremental_cholesky!(B, n) 
#IC = cholesky(A)
vals = zeros(1, n)
for i=1:n
    sublogdet = log(det(A[[1:i-1;i+1:end], [1:i-1;i+1:end]])) 
    vals[i] = sublogdet
    #println("actual sublogdets $i: ", sublogdet)
end
vals1 = zeros(1, n)
logdet_original = log(det(IC))
for i = 1:n
    sublogdet = logdet_loocv_IC(IC, logdet_original, i)
    vals1[i] = sublogdet
    #println("fast sublogdet $i: ", sublogdet)
end
@testset "logdet_loocv_IC" begin
    @test norm(vals-vals1)<1e-10
end


#####
##### test logdet_loocv
#####
#println("test logdet_loocv...")
n=5
#A = rand(n, n); A = A'*A + UniformScaling(1.0)
#println("A: ", A)
val = zeros(1, n)
chol = cholesky(A)
for i=1:n
    sublogdet = log(det(A[[1:i-1;i+1:end], [1:i-1;i+1:end]])) 
    vals[i] = sublogdet
    #println("actual sublogdets $i: ", sublogdet)
end

val1 =zeros(1, n)
logdet_original = log(det(chol))
for i = 1:n
    sublogdet = logdet_loocv(chol, logdet_original, i)
    vals1[i] = sublogdet
    #println("fast sublogdet $i: ", sublogdet)
end

@testset "logdet_loocv" begin
    @test norm(vals-vals1)<1e-10
end

#####
##### test logdet_XΣX_loocv
#####
#println("Running test logdet_XΣX_loocv...")
n = 30
m = 10
X = rand(n, m)
A = (w = rand(n, n);w'*w + UniformScaling(5))
cholA = cholesky(A)
logdetA = logdet(cholA)
bigmat = [A X; X' zeros(m, m)]
#for i = 1:n
#    println("actual subdet $i: ", det(bigmat[[1:i-1;i+1:end], [1:i-1;i+1:end]]))
#end
vals = zeros(1, n)
@time begin
for i = 1:n
    X_minus_i = X[[1:i-1;i+1:end], :]
    sigma_minus_i = A[[1:i-1;i+1:end], [1:i-1;i+1:end]]
    vals[i] = log(abs(det(X_minus_i'*(sigma_minus_i\X_minus_i))))
    #println("actual sublogdet $i: ", log(abs(det(X_minus_i'*(sigma_minus_i\X_minus_i)))))
end
end
vals1 = zeros(1, n)
@time begin
for i =1:n
    sublogdet =  logdet_XΣX_loocv(X, cholA, logdetA, i)
    vals1[i] = sublogdet
    #println("fast sublogdet $i: ", sublogdet)
end
end
println("norm of errors: ", norm(vals-vals1))

@testset "logdet_XΣX_loocv" begin
    @test norm(vals-vals1)<1e-10
end

#####
##### test logdet_XΣX_loocv_IC
#####
#println("Running test logdet_XΣX_loocv_IC...")
n = 30
m = 10
X = rand(n, m)
#A = (w = rand(n, n);w'*w + UniformScaling(5))
B = copy(A) #from previous
cholA = incremental_cholesky!(B, size(A)[1])
logdetA = logdet(cholA)
bigmat = [A X; X' zeros(m, m)]
#for i = 1:n
#    println("actual subdet $i: ", det(bigmat[[1:i-1;i+1:end], [1:i-1;i+1:end]]))
#end
vals = zeros(1, n)
@time begin
for i = 1:n
    X_minus_i = X[[1:i-1;i+1:end], :]
    sigma_minus_i = A[[1:i-1;i+1:end], [1:i-1;i+1:end]]
    vals[i] = log(abs(det(X_minus_i'*(sigma_minus_i\X_minus_i))))
    #println("actual sublogdet $i: ", log(abs(det(X_minus_i'*(sigma_minus_i\X_minus_i)))))
end
end
vals1 = zeros(1, n)
@time begin
for i =1:n
    sublogdet =  logdet_XΣX_loocv_IC(X, cholA, logdetA, i)
    vals1[i] = sublogdet
    #println("fast sublogdet $i: ", sublogdet)
end
end
#println("norm of errors: ", norm(vals-vals1))

@testset "logdet_XΣX_loocv_IC" begin
    @test norm(vals-vals1)<1e-10
end
