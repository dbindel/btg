include("loocv.jl")
#test problem
A = rand(5, 3) 
b = rand(5, 1)
xstar = A\b
r = b-A*xstar
aqr = qr(A)
remainders, minimizers = lsq_loocv(A, aqr, r, xstar) 
rr, mm = lsq_loocv(A, aqr, r, xstar, 3)    

remainders_actual = similar(remainders)
minimizers_actual = similar(minimizers)
for i = 1:size(A, 1)
    A1 = A[[1:i-1;i+1:end], :]
    b1 = b[[1:i-1;i+1:end]]
    x1 = A1\b1
    minimizers_actual[:, i] = x1
    remainders_actual[:, i] = b-A*x1
end
println("remainder error: ", norm(remainders_actual - remainders))
println("minimizer error: ", norm(minimizers_actual - minimizers))

###
### lin_sys_loocv test
###

K = [1.0         0.00369786  0.0608101;
0.00369786  1.0         0.0608101;
0.0608101   0.0608101   1.0]
println("K: ", K)
y = rand(3, 10)
#cholK = incremental_cholesky!(K, 3)

U = cholesky(K).U
c = K\y
#solve subsystem of Kc=y
c1 = lin_sys_loocv(c, U, 1)
c2 = lin_sys_loocv(c, U, 2)
c3 = lin_sys_loocv(c, U, 3)
Cs = zeros(6, 10)
for i = 1:3
    Ki = K[[1:i-1;i+1:end], [1:i-1;i+1:end]]
    Cs[2*i-1:2*i, :] = Ki\y[[1:i-1;i+1:end], :]
end
println("error of indirect error computation: ", norm(Cs - vcat(vcat(c1, c2), c3)))

cholK = incremental_cholesky!(K, 3)
c1IC = lin_sys_loocv_IC(c, cholK, 1)
c2IC = lin_sys_loocv_IC(c, cholK, 2)
c3IC = lin_sys_loocv_IC(c, cholK, 3)
println("error of indirect error computation using Incremental_Cholesky (IC): ", norm(Cs - vcat(vcat(c1IC, c2IC), c3IC)))


###
### Larger test problem. Not activated for now.
###
if true
    n = 50
    K = rand(n, n)
    K = K'*K+UniformScaling(10.0)
    y = rand(n)
    U = cholesky(K).U
    c = K\y
    #solve subsystem of Kc=y
    @time begin 
    indirectCs = zeros(n-1, n)
    for i = 1:n
        indirectCs[:, i] = lin_sys_loocv(c, U, i)
    end
    end
    @time begin
    Cs = zeros(n-1, n)
    for i = 1:n
        Ki = K[[1:i-1;i+1:end], [1:i-1;i+1:end]]
        Cs[:, i] = Ki\y[[1:i-1;i+1:end]]
    end
    end
    println("error of indirect error computation for size $n problem: ", norm(Cs - indirectCs))


    cholK = incremental_cholesky!(K, n)
    indirectCs2 = zeros(n-1, n)
    for i = 1:n
        indirectCs2[:, i] = lin_sys_loocv_IC(c, cholK, i)
    end
    println("error of indirect error computation for size $n problem using Incremental Cholesky: ", norm(Cs - indirectCs2))

end