# lambda: uniform
function prior_pdf_lambda(lambda, lambdamin, lambdamax)
    return 1/(lambdamax - lambdamin)
end

# theta: uniform 
function prior_pdf_theta(theta, thetamin, thetamax)
    l1 = thetamax[1] - thetamin[1]
    l2 = thetamax[2] - thetamin[2]
    return 1/(l1*l2)
end

thetamin = [0, 0]
thetamax = [1, 2]
lambdamin = -3
lambdamax = 3

pdf_lambda = lambda -> prior_pdf_lambda(lambda, lambdamin, lambdamax)
pdf_theta = theta -> prior_pdf_theta(theta, thetamin, thetamax)

param_prior_lambda = (pdf = pdf_lambda, min = lambdamin, max = lambdamax)
param_prior_theta = (pdf = pdf_theta, min = thetamin, max = thetamax)

param_priorInfo = (lambda = param_prior_lambda, theta = param_prior_theta)