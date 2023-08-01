function fba_loop(N, W, u0, warmup_flag=0)
    #instantiate initial times
    deltat = 1/(60*60) #genetic timescale, seconds
    starttime = 0.
    endtime = starttime + deltat
    tspan = [starttime, endtime]
    savetimes = [starttime, endtime] #saving start time and end times for alternate v_dp calculation

    fba_df = DataFrame("v_dp" => [0.0])
    lam = predict(lam_model, fba_df)[1]
    v_in_fpp =  predict(v_in_fpp_model, fba_df)[1]
    v_in_ipp = predict(v_in_ipp_model, fba_df)[1]
    
    p = [lam, v_in_fpp, v_in_ipp, W] 

    #FBA-ODE optimization loop
    ode_data = DataFrame("time" => [0], "fpp" => u0[1], "ipp" => u0[2], "ggp" => u0[3], "phy" => u0[4], "lyc" => u0[5], "bcar" => u0[6], "crtE" => u0[7], "crtB" => u0[8], "crtI" => u0[9], "crtY" => u0[10], "v_dp" => [0], "feas" => [1])
    fba_data = DataFrame("time" => [0], "v_in_fpp" => [v_in_fpp], "v_in_ipp" => [v_in_ipp], "lam" => [lam]);

    #println("Beginning loop...")
    for i in 1:N
        #println("Iteration ", i)
        prob = ODEProblem(beta_carotene, u0, tspan, p)
        #Solve ODE
        sol = solve(prob, Rosenbrock23(), reltol=1e-3, abstol=1e-6, saveat=savetimes)

        #Solve for pathway fluxes from final concentrations
        v_dp = sol.u[end][7] * michaelismenten_dual(sol.u[end][1], sol.u[end][2], bc_params("kcat_crtE"), bc_params("km_crtE_fpp"), bc_params("km_crtE_ipp"))
    
        #Predict if FBA reaction is feasible using ML model
        flux_data = DataFrame("v_dp" => [v_dp])
        feas = predict(feas_model, flux_data)
        feas_class = ifelse.(feas .> 0.5, 1, 0)[1]

        if feas_class == 0 && warmup_flag == 1
            break
        end
        
        ode_data = vcat(ode_data, DataFrame("time" => sol.t[1], "fpp" => sol.u[1][1], "ipp" => sol.u[1][2], "ggp" => sol.u[1][3], "phy" => sol.u[1][4], "lyc" => sol.u[1][5], "bcar" => sol.u[1][6], "crtE" => sol.u[1][7], "crtB" => sol.u[1][8], "crtI" => sol.u[1][9], "crtY" => sol.u[1][10], "v_dp" => v_dp, "feas" => [feas_class]))
        
        flux_data = DataFrame("v_dp" => [v_dp])
        #Predict new v_in, lam using ML model
        lam = predict(lam_model, flux_data)[1]
        v_in_fpp =  predict(v_in_fpp_model, fba_df)[1]
        v_in_ipp = predict(v_in_ipp_model, fba_df)[1]


        p = [lam, v_in_fpp, v_in_ipp, p[4]] 

        starttime = endtime
        endtime = starttime + deltat
        tspan = [starttime, endtime]
        savetimes = [starttime, endtime]

        u0 = sol.u[end]
        #Save FBA data
        fba_data = vcat(fba_data, DataFrame("time" => [starttime], "v_in_fpp" => [p[2]], "v_in_ipp" => [p[3]], "lam" => [p[1]]))
    end
    return ode_data, fba_data
end

function best_in_df(df, col)
    #Helper function to find best row in DF and produce mask
    min = minimum(df[!, col])
    mask = []
    for d in df[!, col]
        if d == min
            push!(mask, 1)
        else push!(mask, 0)
        end 
    end
    return mask
end

###WARMUP ROUTINE
function bayesopt_ics(ic_iters, stable_iters, W, fpp_max = 0.755, ipp_max = 0.755)
    objectives = []
    fpps = []
    ipps = []
    function ic_objective(params)
        fpp, ipp = values(params)
        u0 = [fpp, ipp, 0., 0., 0., 0., 0., 0., 0., 0.]
        ode_data, fba_data = fba_loop(stable_iters, W, u0, 1)
        objective = 1/(fpp + ipp)
        if nrow(ode_data) < stable_iters && nrow(fba_data) < stable_iters
            objective = objective + 10E7
        end
        push!(objectives, objective)
        push!(fpps, fpp)
        push!(ipps, ipp)
        return objective
    end

    space = Dict(
        :fpp => HP.Uniform(:fpp, 0., fpp_max),
        :ipp => HP.Uniform(:ipp, 0., ipp_max))

    best = fmin(ic_objective, space, ic_iters)
    fpp_best, ipp_best = values(best)

    data = DataFrame(:fpp => fpps, :ipp => ipps, :ic_obj => objectives)
    # for (i,fpp, ipp) in ho
    #     bo_data = vcat(bo_data, DataFrame("w1" => W[1], "w2" => W[2], "w3" => W[3], "w4" => W[4], "fpp" => fpp, "ipp" => ipp))
    # end
    data[!, :best_ic] = best_in_df(data, "ic_obj")
    data[!, :w] = fill(W, size(data, 1))

    return fpp_best, ipp_best, minimum(objectives), data
end