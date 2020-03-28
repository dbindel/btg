## Maybe a BO-style test would be like ... 


# build the BTG solver
# allow the user specify choose kernel, transform, quadtype
# INPUT: kernel type, transform type, N (maximum number of points the solver can handle)
btg = BTG(kernel, transform, quadtype, N)

# read some training data: x_train, z_train
# add initial training data 
new_point!(btg, x_train, z_train)

# solve the current model
# equivalent to getTensorGrid in current codebase
pdf, cdf, pdf_deriv = solve(btg)

# increase size of training data
# optimize LCB
x_new = optimize(btg, pdf, cdf, pdf_deriv)

# evaluate and add new point to btg 
# objective function: f
add_point!(btg, x_new, f(x_new))

# solve the current model
pdf, cdf, pdf_deriv = solve(btg)




