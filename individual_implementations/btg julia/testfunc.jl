using Plots

#plt1= plot([1, 2], [3, 4])
#plt2= plot([3, 2], [3, 4], reuse=false)
#@time begin
#display(plot(plt1, plt2))
#end
#display(plot(plt1))
#display(plot(plt2))
#gui()
x=5
y= 10

function f(x=x, y=y)
    return x+y
end
f()