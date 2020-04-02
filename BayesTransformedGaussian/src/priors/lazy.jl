complete_me(obj, h) = (obj.data = h; obj)

struct Lazy
           data
           Lazy() = v -> complete_me(new(), v)
     end