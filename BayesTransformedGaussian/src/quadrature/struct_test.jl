struct a 
    b
    a(x) = new(x)
    a(x, y) = new(x+y)
    function a(x, y, z)
        new(x*y+z)
    end
end

