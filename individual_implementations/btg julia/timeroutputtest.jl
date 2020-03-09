using TimerOutputs

@timeit begin
    3*3
    @timeit 2+2
end