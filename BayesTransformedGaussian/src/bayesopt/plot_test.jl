using PyPlot

#################
#  Create Data  #
#################
x = 100*rand(50)
y = 100*rand(50)
areas = 800*rand(50)

##################
#  Scatter Plot  #
##################
fig = figure("pyplot_scatterplot",figsize=(10,10))
ax = PyPlot.axes()
scatter(x,y,s=areas,alpha=0.5)

PyPlot.title("Scatter Plot")
xlabel("X")
ylabel("Y")
grid("on")