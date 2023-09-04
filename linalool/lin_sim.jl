function fba_loop_no_ml(N, W, u0, warmup_flag=0)
    #instantiate initial times
    deltat = 1/(60*60) #genetic timescale, seconds
    starttime = 0.
    endtime = starttime + deltat
    tspan = [starttime, endtime]
    savetimes = [starttime, endtime] #saving start time and end times for alternate v_dp calculation

    fba_df = DataFrame("v_dp" => [0.0])
    
end

