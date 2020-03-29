#Julia allows you to define functions on structs

struct y
    a 
end

(in::y)(b) = in.a+b

p = y(3)
p(4)