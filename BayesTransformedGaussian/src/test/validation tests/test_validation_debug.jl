using Dates
#using ProfileView

include("../btg.jl")
include("../datasets/load_abalone.jl")

#ind = 350:370
ind = 3300:3359
#posx = 1:3 #
posx = [1;4]
posc = 1:1
x = data[ind, posx] 
#choose a subset of variables to be regressors for the mean
Fx = data[ind, posc] 
y = float(target[ind])
pind = 10:10 #prediction index
#pind = i:i
trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions
#prediction data
d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
Fx0 = reshape(data[pind, posc], 1, length(posc))
x0 = reshape(data[pind, posx], 1, length(posx)) 
#rangeθ = [100.0 200]
rangeθ = [100.0 200]
rangeλ = [-1.0 1] #we will always used 1 range scale for lambda
btg1 = btg(trainingData1, rangeθ, rangeλ; quadtype = ["MonteCarlo", "MonteCarlo"])
#θ1 = btg1.nodesWeightsθ.nodes[1, 6] #pick some theta value which doubles as quadrature node
#λ1 = btg1.nodesWeightsλ.nodes[3]
##################################################################################################
PyPlot.close("all") #close existing windows

#(pdf, cdf, dpdf) = solve(btg1); #initialize training_buffer_dicts, solve once so can use fast techiques to extrapolate submatrix determinants, etc.
#pdf(x0, Fx0, .4);
#(pdf, cdf, dpdf) = solve(btg1; validate = i) #do LOOCV this time around
#a1 = y-> pdf(x0, Fx0, y)

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
"""
Run
"""
function test_LOOCV(btg1::btg, m::Int64, n::Int64; fast)
    fast = fast
    m = m; n=n

    #plt, axs = PyPlot.subplots(m, n,figsize=[20,11])
    plt, axs = PyPlot.subplots(m, n)

    #figure(1)
    before = Dates.now()
    println("LOOCV type is fast?: $fast")
    for j = 8:8
        ind1 = Int64(ceil(j/n))
        ind2 = Int64(j - n*(floor((j-.1)/n)))
        println("iteration $j")
        if j <m*n
            if fast
                (pdf, cdf, dpdf) = solve(btg1, validate = j);   
                b1 = y0 -> pdf(x0, Fx0, y0);
                c1 = y0 -> cdf(x0, Fx0, y0);
                (x, y) = plt_data(b1, .01, 1.2, 100)
                (xc, yc) = plt_data(c1, .01, 1.2, 100)
            else
                (trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData1, j)
                btg2 = btg(trainingdata_minus_i, rangeθ, rangeλ; quadtype = ["MonteCarlo", "MonteCarlo"])
                (pdf_minus_i, cdf_minus_i, dpdf_minus_i) = solve(btg2)
                b1 = y -> pdf_minus_i(x_i, Fx_i, y) 
                c1 = y -> cdf_minus_i(x_i, Fx_i, y) 
                (x, y) = plt_data(b1, .01, 1.2, 100)
                (xc, yc) = plt_data(c1, .01, 1.2, 100)
            end
            axs[ind1, ind2].plot(x, y, color = "red", linewidth = 2.0, linestyle = ":")
            axs[ind1, ind2].plot(xc, yc, color = "orange", linewidth = 2.0, linestyle = ":")
            #axs[ind1, ind2].plot(x1, y1, color = "blue", linewidth = 1.0, linestyle = "--")
            axs[ind1, ind2].axvline(x =  getLabel(btg1.trainingData)[j])
        else #do annotation in last box
            #println("annotating...")
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
end

test_LOOCV(btg1, 6, 10; fast = true)

