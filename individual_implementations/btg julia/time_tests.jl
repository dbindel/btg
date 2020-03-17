using LinearAlgebra
using Distances, StaticArrays, StatsFuns

if true 
x = rand(10000, 100000)
x = x'*x+UniformScaling(1.0)
end

#cholesky 10000x10000 takes 5-6 sec
# matmul two 100000 matrices takes 30 secr
# matvec takes 0.1 sec
#cholesky solve 0.15 sec

#forming kernel matrix takes 2.217 seconds when s is 1D, 3 seconds when s is 7D
#0.5 seconds to take dot product of 10,000^2 matrix with another matrix
#.4 seconds to square entries of 10,000^2 matrix

preallocated = zeros(10000,10000)

#15000x15000 case
#x*x 50 sec O(n^3)
#x*z 0.2 sec
# x hadamard product x: 1 second 
#cholesky(x) solve: .3 seconds
#cholesky: 17 seconds O(n^3)

# ------------takeaway here is the math adds up-------------
#fastK(s, s):  7 seconds 
#pairwise(s, s): 2 seconds
#exponentiate the result^: 2seconds 
#divide by pi: 1.2 seconds

#0 pairwise: 2 seconds
#1.squaring operation ~1 second
#2. scale by - theta operation ~1 second
# 3. exponentiate operation ~2seconds
#divide by constant operation ~1 sec


#20000 x 20000 case
#fastK:  17 - 33 sec
#cholesky: 61 sec
#cholesky solve: 0.5 sec
#matvec with kernel matrix: 0.3 sec
#dot product k .* k: 1.3 sec

