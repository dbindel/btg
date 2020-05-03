 using Plots
 pdf_fixed(v) = pdf(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
 cdf_fixed(v) = cdf(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
 cdf_gradient_fixed(v) = cdf_gradient(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
 cdf_hessian_fixed(v) = cdf_hessian(v[2:end], linear_polynomial_basis(v[2:end]), v[1])

pdf_vstar = y -> pdf_fixed(vcat(y, vstar[2:end]))
cdf_vstar = y -> cdf_fixed(vcat(y, vstar[2:end]))
#plt(cdf_vstar, .0001, 1000.0)
#plt(pdf_vstar, .0001, 50)

