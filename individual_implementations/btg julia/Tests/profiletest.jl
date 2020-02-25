using Profile
using ProfileView
function myfunc()
           A = rand(200, 200, 400)
           maximum(A)
       end

myfunc()
@profile myfunc()