"""
Plots history of sampled points or init_points
"""
function visualize_search(hist, color = "blue")
    n = length(hist)
    xx = zeros(1, n)
    yy = zeros(1, n)
    
    for i = 1:n
        xx[i] = hist[i][1]
    end
    for i = 1:n
        yy[i] = hist[i][2]
    end
    Plots.scatter!(xx, yy, color = color, legend = false)
end

function visualize_search2(hist, color = "red")
    n = size(hist, 2)
    xx = zeros(1, n)
    yy = zeros(1, n)
    
    for i = 1:n
        xx[i] = hist[i][1]
    end
    for i = 1:n
        yy[i] = hist[i][2]
    end
    Plots.scatter!(xx, yy, color = color, legend = false)
end
visualize_search(init_hist, "red")
visualize_search(x_hist)
#labels = ["1", "2", "3"]

#for i = 1:25
#    init_hist[i] = init_hist[i][2:end]
#end

