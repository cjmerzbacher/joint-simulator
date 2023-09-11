function fba_loop(N, params, u0, warmup_flag=0)
    #instantiate initial times
    deltat = 1/(60*60) #genetic timescale, seconds
    starttime = 0.
    endtime = starttime + deltat
    tspan = [starttime, endtime]
    savetimes = [starttime, endtime] #saving start time and end times for alternate v_dp calculation

    fba_df = DataFrame("v_dp" => [0.0])
    lam = predict(lam_model, fba_df)[1]
    v_in = v_in_model(fba_df.v_dp')[1]
    A, W = params
    p = [A, W, v_in, lam] 

    #FBA-ODE optimization loop
    ode_data = DataFrame("time" => [0], "g6p" => u0[1], "f6p" => u0[2], "mi" => u0[3], "ino1" => u0[4], "miox" => u0[5], "v_dp" => [0], "v_out" => [0], "feas" => [1])
    fba_data = DataFrame("time" => [0], "v_in" => [v_in], "lam" => [lam]);

    for i in 1:N
        prob = ODEProblem(glucaric_acid, u0, tspan, p)
        #Solve ODE
        sol = solve(prob, Rosenbrock23(), reltol=1e-3, abstol=1e-6, saveat=savetimes)

        #Solve for pathway fluxes from final concentrations
        v_dp = sol.u[end][4] * michaelismenten(sol.u[end][1], ga_params("vm_ino1"), ga_params("km_ino1_g6p")) #same as v_ino1 = ino1*mm(g6p, vm, km)    
        v_out = hillequation(sol.u[end][2], ga_params("vm_pfk"), ga_params("n_pfk"), ga_params("km_pfk_f6p"))
        
        #Predict if FBA reaction is feasible using ML model
        flux_data = DataFrame("v_dp" => [v_dp])
        feas = predict(feas_model, flux_data)
        feas_class = ifelse.(feas .> 0.5, 1, 0)[1]

        if feas_class == 0 && warmup_flag == 1
            break
        end
        
        ode_data = vcat(ode_data, DataFrame("time" => sol.t[1], "g6p" => sol.u[1][1], "f6p" => sol.u[1][2], "mi" => sol.u[1][3], "ino1" => sol.u[1][4], "miox" => sol.u[1][5], "v_dp" => v_dp, "v_out" => v_out, "feas" => [feas_class]))        
        
        flux_data = DataFrame("v_dp" => [v_dp])
        #Predict new v_in, lam using ML model
        lam = predict(lam_model, flux_data)[1]
        v_in = v_in_model(flux_data.v_dp')[1]

        p = [A, W, v_in, lam] 

        starttime = endtime
        endtime = starttime + deltat
        tspan = [starttime, endtime]
        savetimes = [starttime, endtime]

        u0 = sol.u[end]
        #Save FBA data
        fba_data = vcat(fba_data, DataFrame("time" => [starttime], "v_in" => [p[3]], "lam" => [p[4]]))    
    end
    return ode_data, fba_data
end


     
###WARMUP ROUTINE
function warmup(num_iters, W_A, f6p_max = 0.281, g6p_max = 0.068)
    objectives = []
    f6ps = []
    g6ps = []

    function warmup_objective(params)
        f6p, g6p = values(params)
        u0 = [g6p, f6p, 0., 0., 0.]
        stable_iters = 10
        ode_data, fba_data = fba_loop(stable_iters, W_A, u0, 1)
        objective = 1/(g6p + f6p)
        if nrow(ode_data) < stable_iters && nrow(fba_data) < stable_iters
            objective = objective + 10E7
        end
        push!(objectives, objective)
        push!(f6ps, f6p)
        push!(g6ps, g6p)
        return objective
    end
    
    space = Dict(
        :f6p => HP.Uniform(:f6p, 0., f6p_max),
        :g6p => HP.Uniform(:g6p, 0., g6p_max))

    best = fmin(warmup_objective, space, num_iters)
    f6p_best, g6p_best = values(best)

    data = DataFrame(:f6p => f6ps, :g6p => g6ps, :obj => objectives)
    
    return f6p_best, g6p_best, minimum(objectives), data
end