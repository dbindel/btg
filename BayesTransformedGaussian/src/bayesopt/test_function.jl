using Plots; 
pyplot()
x=range(-2,stop=2,length=100)
y=range(sqrt(2),stop=2,length=100)
f(x,y) = x*y-x-y+1
plot(x,y,f,st=:surface,camera=(-30,30))