 using Plots
 pdf_fixed(v) = pdf(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
 cdf_fixed(v) = cdf(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
 cdf_gradient_fixed(v) = cdf_gradient(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
 cdf_hessian_fixed(v) = cdf_hessian(v[2:end], linear_polynomial_basis(v[2:end]), v[1])

pdf_vstar = y -> pdf_fixed(vcat(y, vstar[2:end]))
cdf_vstar = y -> cdf_fixed(vcat(y, vstar[2:end]))
#plt(cdf_vstar, .0001, 1000.0)
#plt(pdf_vstar, .0001, 50)

function vcdfplot(vstar; lower = 0.0, upper=1000)
    cdf_vstar = y -> cdf_fixed(vcat(y, vstar[2:end]))
    plt(cdf_vstar, lower, upper)
end

"""
Multiple graphs per plot
"""
function vcdfplot_sequence(vstars; lower = 0.0, upper = 1000, res = nothing)
    cdf_vstar = y -> cdf_fixed(vcat(y, vstar[2:end]))
    Plots.plot()
    for i = 1:length(vstars)
        vstar = vstars[i]
        cdf_vstar = y -> cdf_fixed(vcat(y, vstar[2:end]))
        if res!=nothing
            #label = abs(res[i]-0.25) < .01 ? "converged" : "did not converge" 
            color = abs(res[i]-0.25) < .01 ? "blue" : "red"
            plt!(cdf_vstar, lower, upper; color = color)
        end
    end
end

