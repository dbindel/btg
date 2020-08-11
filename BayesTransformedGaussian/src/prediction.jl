using TimerOutputs
if !@isdefined(to)
    const to = TimerOutput()
end

mutable struct btgResults
    time::Dict{String, T} where T<:Real
    data::Dict{String, Array}
    function btgResults()
        time = Dict(
                    "time_median" => 0., 
                    "time_CI" => 0., 
                    "time_eval" => 0.,
                    "time_preprocess" => 0.,
                    "time_total" => 0.
                    )
        data = Dict(
                    "pdf" => Array{Function, 1}(), 
                    "cdf" => Array{Function, 1}(), 
                    "dpdf" => Array{Function, 1}(), 
                    "median" => Array{Real, 1}(), 
                    "credible_intervel" => Array{Array{Real}, 1}(),
                    "absolute_error"=> Array{Real, 1}(),
                    "squared_error"=> Array{Real, 1}(),
                    "negative_log_predictve_density"=> Array{Real, 1}()
                    ) 
        new(time, data)
    end
end

"""
merge new prediction results into exisiting results
"""
function merge_results!(results::btgResults, results_new::btgResults)
    @assert keys(results.time) == keys(results_new.time) "Results should share same time key"
    @assert keys(results.data) == keys(results_new.data) "Results should share same data key"
    for key in keys(results.time)
        results.time[key] += results_new.time[key]
    end
    for key in keys(results.data)
        append!(results.data[key], results_new.data[key])
    end
end

"""
btgPredict object contains prediction results, including
"""
mutable struct btgPredict
    testingdata::AbstractTestingData 
    pdf::Array{Function, 1}
    cdf::Array{Function, 1}
    dpdf::Array{Function, 1} 
    median::Array{Real, 1} 
    credible_intervel::Array{Array, 1}
    absolute_error::Union{Array{T, 1}, Nothing} where T<:Real
    squared_error::Union{Array{T, 1}, Nothing} where T<:Real
    negative_log_pred_density::Union{Array{T, 1}, Nothing} where T<:Real
    time_cost::Dict{String, T} where T<:Real
    debug_log::Any
    function predict_single(x_i::Array{T,2}, Fx_i::Array{T,2}, 
                            pdf_raw::Function, cdf_raw::Function, dpdf_raw::Function, 
                            quantInfo_raw::Function, ymax::T; 
                            y_i_true::T = nothing, confidence_level = .95) where T<:Real
        results_i = btgResults() # initialize a results object for the i-th point
        @timeit to "time_total" begin
            @timeit to "time_preprocess" pdf_i, cdf_i, dpdf_i, quantbound_i, support_i, int_i = pre_process(x_i, Fx_i, pdf_raw, cdf_raw, dpdf_raw, quantInfo_raw)
            @timeit to "time_median" median_i = ymax * quantile(cdf_i, quantbound_i, support_i)[1]
            @timeit to "time_CI" CI_i = ymax .* credible_interval(cdf_i, quantbound_i, support_i; mode=:equal, wp=confidence_level)[1]
            append!(results_i.data["pdf"], [pdf_i])
            append!(results_i.data["cdf"], [cdf_i])
            append!(results_i.data["dpdf"], [dpdf_i])
            append!(results_i.data["median"], [median_i])
            append!(results_i.data["credible_intervel"], [CI_i])
            if y_i_true != nothing
                @timeit to "time_eval" begin
                    append!(results_i.data["absolute_error"], [abs(y_i_true - median_i)])
                    append!(results_i.data["squared_error"], [(y_i_true - median_i)^2])
                    append!(results_i.data["negative_log_predictve_density"], [-log(pdf_i(y_i_true/ymax))])
                end
            end
        end
        results_i.time["time_preprocess"] += TimerOutputs.time(to["time_total"]["time_preprocess"])/1e9
        results_i.time["time_median"] += TimerOutputs.time(to["time_total"]["time_median"])/1e9
        results_i.time["time_CI"] += TimerOutputs.time(to["time_total"]["time_CI"])/1e9
        results_i.time["time_eval"] += TimerOutputs.time(to["time_total"]["time_eval"])/1e9
        results_i.time["time_total"] += TimerOutputs.time(to["time_total"])/1e9
        debug_log = nothing
        return results_i, debug_log
    end

    function btgPredict(x::Array{T,2}, Fx::Array{T,2}, btg::btg; y_true::Array{T,2} = nothing) where T<:Real
        confidence_level = btg.options.confidence_level
        testingdata = testingData(x_test, Fx_test; y0_true = y_true)
        ymax = btg.trainingData.ymax
        n_test = getNumPts(testingdata)
        dimx = getDimension(testingdata)
        dimFx = getCovDimension(testingdata)
        btgmodel = btgModel(btg); # previous solve function
        pdf_raw = btgmodel.pdf
        cdf_raw = btgmodel.cdf
        dpdf_raw = btgmodel.dpdf
        quantInfo_raw = btgmodel.quantInfo
        results = btgResults() # initialize result for testing data
        debug_log = []
        # predict one by one, could put in parallel
        for i in 1:n_test
            x_i = reshape(x[i, :], 1, dimx)
            Fx_i = reshape(Fx[i, :], 1, dimFx)
            y_i_true = y_true == nothing ? nothing : y_true[i] 
            results_i, debug_log_i = predict_single(x_i, Fx_i, pdf_raw, cdf_raw, dpdf_raw, quantInfo_raw, ymax; y_i_true = y_i_true, confidence_level = confidence_level)
            # merge results_i into results
            merge_results!(results, results_i)
            # append!(debug_log, debug_log_i)
        end
        # unpack results
        pdf = results.data["pdf"]
        cdf = results.data["cdf"]
        dpdf = results.data["dpdf"]
        median = results.data["median"]
        credible_intervel = results.data["credible_intervel"]
        time_cost = results.time
        if y_true != nothing # evaluate results
            absolute_error = results.data["absolute_error"]
            squared_error = results.data["squared_error"]
            negative_log_pred_density = results.data["negative_log_predictve_density"]
        else
            absolute_error = nothing
            squared_error = nothing
            negative_log_pred_density = nothing
        end
        new(testingdata, pdf, cdf, dpdf, median, credible_intervel, absolute_error, squared_error, negative_log_pred_density, time_cost, debug_log)
    end
end




