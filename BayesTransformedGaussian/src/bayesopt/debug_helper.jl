"""
Plots history of sampled points or init_points
"""
function visualize_search(hist, color = "blue")
    xx = zeros(1, 25)
    yy = zeros(1, 25)
    
    for i = 1:25
        xx[i] = hist[i][1]
    end
    for i = 1:25
        yy[i] = hist[i][2]
    end
    Plots.scatter!(xx, yy, color = color)
end

function visualize_search2(hist, color = "blue")
    xx = zeros(1, 25)
    yy = zeros(1, 25)
    
    for i = 1:25
        xx[i] = hist[i, 1]
    end
    for i = 1:25
        yy[i] = hist[i,2]
    end
    Plots.scatter!(xx, yy, color = color)
end

labels = ["1", "2", "3"]

for i = 1:25
    init_hist[i] = init_hist[i][2:end]
end

