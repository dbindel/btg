include("../btg.jl")
include("../bayesopt/updateBTG.jl")
using Test

# Test updateBTG.jl in bayesopt folder
# tests if trainingData, testingData, and 3 kinds of buffers are upadated correctly
#

x = [0.5 0.1; 0.15 0.4; 0.4 0.6; 0.2 0.3];
Fx = linear_polynomial_basis(x);
y = [1.0, 3.0, 1.5, 2.6];
x0 = [0.8 0.35];
Fx0 = linear_polynomial_basis(x0);
range_theta = [1.0 5];
range_lambda = [2.0 4]
train_data = extensible_trainingData(x, Fx, y);
test_data = testingData(x0, Fx0);
b1 = btg(train_data, range_theta, range_lambda);
x_new = [.2 .7]
Fx_new = linear_polynomial_basis(x_new)
y_new = [1.6]

lambda_keys = keys(b1.λbuffer_dict)
theta_keys = keys(b1.train_buffer_dict)
arb_lambda_key = first(lambda_keys)
arb_theta_key = first(theta_keys)

randb = rand(5,1)

test_data = testingData(x0, Fx0)

#println(b1.train_buffer_dict[arb_theta_key].choleskyΣθ \ b])

updateBTG!(b1, x_new, Fx_new, y_new) #should update trainingData, and all three buffers



if true #test set
@testset "trainingData update test" begin
    @test b1.trainingData.x[b1.trainingData.n:b1.trainingData.n, :] == x_new
    @test b1.trainingData.Fx[b1.trainingData.n:b1.trainingData.n, :] == Fx_new
    @test [b1.trainingData.y[b1.trainingData.n]] == y_new
end

@testset "λbuffer update test" begin
    g = x -> b1.g(x, arb_lambda_key)
    dg = x -> partialx(b1.g, x, arb_lambda_key)
    #@info b1.λbuffer_dict[arb_lambda_key].dgλz[end]
    #@info dg(y_new)
    #@info b1.λbuffer_dict[arb_lambda_key].gλz[end]
    #@info y_new
    @test [b1.λbuffer_dict[arb_lambda_key].dgλz[end]] == dg(y_new)
    @test [b1.λbuffer_dict[arb_lambda_key].gλz[end]] == g(y_new) 
end

@testset "train_buffer update test" begin
    x = getPosition(b1.trainingData)
    Fx = getCovariates(b1.trainingData)
    chol = cholesky( correlation(Gaussian(), arb_theta_key, x) + 1e-12 * Matrix(I, 5, 5)) #assume fixed noise level of 1e-12
    #@info chol\randb
    #@info b1.train_buffer_dict[arb_theta_key].choleskyΣθ\randb
    @test norm(chol\randb - b1.train_buffer_dict[arb_theta_key].choleskyΣθ\randb) <1e-6

    @test norm(b1.train_buffer_dict[arb_theta_key].Σθ_inv_X - chol\Fx) <1e-6
    v = rand(3, 1)
    @test norm(b1.train_buffer_dict[arb_theta_key].choleskyXΣX\v  - (Fx' * (chol\Fx))\v)<1e-6
    @test norm(b1.train_buffer_dict[arb_theta_key].logdetΣθ - logdet(chol))<1e-6
    @test norm(b1.train_buffer_dict[arb_theta_key].logdetXΣX - logdet(Fx' * (chol\Fx)))<1e-6
end

#@testset "θλbuffer update test"

println("test solve b1...")
(pdf, cdf, dpdf, augmented_cdf_deriv, augmented_cdf_hess, quantInfo, tgridpdf, tgridcdf, tgridm, tgridsigma_m) = solve(b1);
pdf(x0, Fx0, 2) #implicitly initializes b1.testingData

#println(b1.testingData.x0)
#println(b1.testingData.Fx0)
#println(b1.test_buffer_dict[arb_theta_key].Bθ)

@info norm(Bθ - Bθ_actual)

@testset "test_buffer update tests" begin
    Bθ = b1.test_buffer_dict[arb_theta_key].Bθ
    Eθ = b1.test_buffer_dict[arb_theta_key].Eθ
    Dθ = b1.test_buffer_dict[arb_theta_key].Dθ
    Hθ = b1.test_buffer_dict[arb_theta_key].Hθ

    Bθ_actual = cross_correlation(Gaussian(), arb_theta_key, getPosition(b1.testingData), getPosition(b1.trainingData))
    Eθ_actual = [1]
    Dθ_actual = Eθ_actual - Bθ_actual * (b1.train_buffer_dict[arb_theta_key].choleskyΣθ \Bθ_actual')
    Hθ_actual = getCovariates(b1.testingData) - Bθ_actual * b1.train_buffer_dict[arb_theta_key].Σθ_inv_X

    @test norm(Bθ - Bθ_actual) <1e-6
    @test norm(Eθ - Eθ_actual) <1e-6
    @test norm(Dθ - Dθ_actual) <1e-6
    @test norm(Hθ - Hθ_actual) <1e-6
end
if false
#@testset "test multiple btg update" begin
    (pdf, cdf, dpdf, augmented_cdf_deriv, augmented_cdf_hess, quantInfo, tgridpdf, tgridcdf, tgridm, tgridsigma_m) = solve(b1);
    x_new = [1.3 .7]
    y_new = [.6]
    @info "cdf(x_new, linear_polynomial_basis(x_new), .4)" cdf(x_new, linear_polynomial_basis(x_new), .4)
    @info "cdf([.5 .9], linear_polynomial_basis([.5 .9]), .3)" cdf([.5 .9], linear_polynomial_basis([.5 .9]), .3)
    updateBTG!(b1, x_new, linear_polynomial_basis(x_new), y_new)
    (pdf, cdf, dpdf, augmented_cdf_deriv, augmented_cdf_hess, quantInfo, tgridpdf, tgridcdf, tgridm, tgridsigma_m) = solve(b1);
    @info "cdf(x_new, linear_polynomial_basis(x_new), .4)" cdf(x_new, linear_polynomial_basis(x_new), .41)
    @info "cdf([.5 .9], linear_polynomial_basis([.5 .9]), .3)" cdf([.5 .9], linear_polynomial_basis([.5 .9]), .3)
#end

end