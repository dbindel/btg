

include("../btg.jl")
include("../datasets/load_abalone.jl")
i = 3 #LOOCV at this point

ind = 350:370
posx = 1:3 #
posc = 1:1
x = data[ind, posx] 
#choose a subset of variables to be regressors for the mean
Fx = data[ind, posc] 
y = float(target[ind])
#pind = 10:10 #prediction index
pind = i:i
trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions
#prediction data
d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
Fx0 = reshape(data[pind, posc], 1, length(posc))
x0 = reshape(data[pind, posx], 1, length(posx)) 
#rangeθ = [100.0 200]
rangeθ = [10.0 2000]
rangeλ = [-1.0 1] #we will always used 1 range scale for lambda
btg1 = btg(trainingData1, rangeθ, rangeλ) #quadtype = ["SparseGrid", "MonteCarlo"])
#θ1 = btg1.nodesWeightsθ.nodes[1, 6] #pick some theta value which doubles as quadrature node
#λ1 = btg1.nodesWeightsλ.nodes[3]
##################################################################################################
PyPlot.close("all") #close existing windows


function perturb(tup::Tuple{Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 1}})
    x, Fx, y = tup
    x = x + randn(size(x, 1), size(x, 2)) * 0.1
    Fx = Fx + randn(size(Fx, 1), size(Fx, 2)) * 0.1
    y = y + randn(length(y)) * 0.1 
    return (x, Fx, y)
end

(pdf, cdf, dpdf) = solve(btg1); #initialize training_buffer_dicts, solve once so can use fast techiques to extrapolate submatrix determinants, etc.
#pdf(x0, Fx0, .4);
(pdf, cdf, dpdf) = solve(btg1; validate = i) #do LOOCV this time around
a1 = y-> pdf(x0, Fx0, y)

"""
leave-one-out-training-data
"""
function lootd(td::AbstractTrainingData, i::Int64)
    x = getPosition(td)
    Fx = getCovariates(td)
    z = getLabel(td)
    x_minus_i = x[[1:i-1;i+1:end], :]
    Fx_minus_i = Fx[[1:i-1;i+1:end], :]
    z_minus_i = z[[1:i-1;i+1:end]]
    x_i = x[i:i, :]
    Fx_i = Fx[i:i, :]
    z_i = z[i:i, :]
    return trainingData(x_minus_i, Fx_minus_i, z_minus_i), x_i, Fx_i, z_i
end

#btg 2, trained on subset of trainingData
(trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData1, i)
btg2 = btg(trainingdata_minus_i, rangeθ, rangeλ)
(pdf_minus_i, cdf_minus_i, dpdf_minus_i) = solve(btg2)
a2 = y -> pdf_minus_i(x_i, Fx_i, y) 

@assert equals(btg2.θλbuffer_dict, btg1.validation_θλ_buffer_dict) #do the computed buffers match up?

if false #plot
figure(5)
(x1, y1) = plt_data(a1, .01, 1.5, 150)
(x2, y2) = plt_data(a2, .01, 1.5, 150)
PyPlot.plot(x1, y1, color = "blue")
PyPlot.plot(x2, y2, color = "green")
end


if false 
    #function calls to initialize debug_log
    a1(.5);
    a2(.5);

    h1 = sort(btg1.debug_log, by = first, rev = true)[1:5]
    h2 = sort(btg2.debug_log, by = first, rev = true)[1:5]

    f1 = sort(btg1.debug_log, by = last, rev = true)[1:5]
    f2 = sort(btg2.debug_log, by = last, rev = true)[1:5]

end

i=3
(pdf, cdf, dpdf) = solve(btg1; validate = i) #do LOOCV this time around
a3 = y-> pdf(x0, Fx0, y) #doesnt matter what x0 and Fx0 are

#btg 2, trained on subset of trainingData
(trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData1, i)
btg3 = btg(trainingdata_minus_i, rangeθ, rangeλ)
(pdf_minus_i, cdf_minus_i, dpdf_minus_i) = solve(btg3)
a4 = y -> pdf_minus_i(x_i, Fx_i, y) 


if false
    #function calls to initialize debug_log
    a3(.5);
    a4(.5);
    (histx, histy) = plt_data(a3, .01, 1.2, 100);
    (histx2, histy2) = plt_data(a4, .01, 1.2, 100);
    
    figure(27)
    PyPlot.plot(histx, histy)
    Pyplot.plot(histx2, histy2)

    b1 = sort(btg1.debug_log, by = first, rev = true)[1:5]
    b2 = sort(btg3.debug_log, by = first, rev = true)[1:5]

    n1 = sort(btg1.debug_log, by = last, rev = true)[1:5]
    n2 = sort(btg3.debug_log, by = last, rev = true)[1:5]

end

###
###
### plotting stuff
###
###

m = 4; n = 5
plt, axs = PyPlot.subplots(m, n)
#figure(1)
for j = 1:m*n
    (pdf, cdf, dpdf) = solve(btg1, validate = j);  
    #a = y0 -> dpdf(x0, Fx0, y0); 
    b1 = y0 -> pdf(x0, Fx0, y0);
    #c1 = y0 -> cdf(x0, Fx0, y0);
    (x, y) = plt_data(b1, .01, 1.2, 100)

    (trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData1, j)
    btg2 = btg(trainingdata_minus_i, rangeθ, rangeλ)
    (pdf_minus_i, cdf_minus_i, dpdf_minus_i) = solve(btg2)
    b2 = y -> pdf_minus_i(x_i, Fx_i, y) 
    (x1, y1) = plt_data(b2, .01, 1.2, 100)

    ind1 = Int64(ceil(j/n))
    ind2 = Int64(j - n*(floor((j-.1)/n)))

    axs[ind1, ind2].plot(x, y, color = "red", linewidth = 4.0, linestyle = ":")
    axs[ind1, ind2].plot(x1, y1, color = "blue", linewidth = 1.0, linestyle = "--")
    axs[ind1, ind2].axvline(x =  getLabel(btg1.trainingData)[j])
    #PyPlot.plot(x, y)
    #PyPlot.plot(x1, y1)
    #PyPlot.axvline(x =  getLabel(btg1.trainingData)[j])
end
for ax in axs
    ax.set(xlabel="x-label", ylabel="y-label")
end
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs
    ax.label_outer()
end