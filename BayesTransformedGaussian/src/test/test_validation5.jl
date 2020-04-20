using Dates
using ArgParse

include("../btg.jl")
include("../datasets/load_abalone.jl")

s = ArgParseSettings()
@add_arg_table! s begin
    "--fast"
        help = "use fast or not"
        action = :store_true
end
parsed_args = parse_args(ARGS, s)


ind = 3000:3100
posx = 1:7
posc = 1:3
x = data[ind, posx] 
Fx = data[ind, posc] 
y = float(target[ind])

#prediction data
pind = 1:1
Fx0 = reshape(data[pind, posc], 1, length(posc))
x0 = reshape(data[pind, posx], 1, length(posx)) 

#parameter setting
rangeθ = [10.0 1000]
rangeλ = [0. 3.] #we will always used 1 range scale for lambda

trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions
d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
btg1 = btg(trainingData1, rangeθ, rangeλ; quadtype = ["SparseGrid", "Gaussian"])
(pdf, cdf, dpdf) = solve(btg1); #initialize training_buffer_dicts, solve once so can use fast techiques to extrapolate submatrix determinants, etc.


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
###
###
### plotting 
###
###
nrow=6; ncol=8
PyPlot.close("all") #close existing windows
plt, axs = PyPlot.subplots(nrow, ncol)
PyPlot.suptitle("Cross Validation $fast", fontsize=10)
#figure(1)
before = Dates.now()
for j = 1:nrow*ncol
    ind1 = Int64(ceil(j/ncol))
    ind2 = Int64(j - ncol*(floor((j-.1)/ncol)))
    println("iteration $j")
    if j <nrow*ncol
        if parsed_args.fast
            (pdf, cdf, dpdf) = solve(btg1, validate = j);   
            b1 = y0 -> pdf(x0, Fx0, y0);
            c1 = y0 -> cdf(x0, Fx0, y0);
            (x, y) = plt_data(b1, .01, 1.2, 100)
            (xc, yc) = plt_data(c1, .01, 1.2, 100)
        else
            (trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData1, j)
            btg2 = btg(trainingdata_minus_i, rangeθ, rangeλ; quadtype = ["SparseGrid", "Gaussian"])
            (pdf_minus_i, cdf_minus_i, dpdf_minus_i) = solve(btg2)
            b1 = y -> pdf_minus_i(x_i, Fx_i, y) 
            c1 = y -> cdf_minus_i(x_i, Fx_i, y) 
            (x, y) = plt_data(b1, .01, 1.2, 100)
            (xc, yc) = plt_data(c1, .01, 1.2, 100)
        end
        axs[ind1, ind2].plot(x, y, color = "red", linewidth = 1.0, linestyle = "-")
        axs[ind1, ind2].plot(xc, yc, color = "orange", linewidth = 1.0, linestyle = "-")
        axs[ind1, ind2].axvline(x =  getLabel(btg1.trainingData)[j])
    else #do annotation in last box
        after = Dates.now()
        elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
        qtype =  btg1.quadType
        axs[ind1, ind2].annotate(
        L"$\theta: $" * "$rangeθ" * "\n" *
        L"$\lambda: $" * "$rangeλ" * "\n" *
        "$qtype \n" * 
        "ind: " * "$ind \n" * 
        "x: " * "$posx " *", cov: " * "$posc\n" * 
        "time: " * "$elapsedmin", 
        xy=[.03;.03],
        fontsize = 5.0)
        # *
        #L"$\lambda: $" * "$rangeλ" * ".\n" *
        #"quad: " * "$btg1.quadType",
    end
end
#for ax in axs
#    ax.set(xlabel="x-label", ylabel="y-label")
#end
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs
    ax.label_outer()
end

PyPlot.savefig("figure/test_validation5_$(parsed_args.fast).pdf")