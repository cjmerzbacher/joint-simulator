function fba_loop(N, W, u0, warmup_flag=0, models=[feas_model, lam_model, v_in_model, v_fpp_model, v_ipp_model])
    feas_model, lam_model, v_in_model, v_fpp_model, v_ipp_model = models
    #instantiate initial times
    deltat = 1/(60*60) #genetic timescale, seconds
    starttime = 0.
    endtime = starttime + deltat
    tspan = [starttime, endtime]
    savetimes = [starttime, endtime] #saving start time and end times for alternate v_p calculation

    fba_df = DataFrame("v_p" => [0.0])
    lam = predict(lam_model, fba_df)[1]
    v_in =  predict(v_in_model, fba_df)[1]
    v_fpp =  predict(v_fpp_model, fba_df)[1]
    v_ipp = predict(v_ipp_model, fba_df)[1]
    
    p = [lam, v_in, v_fpp, v_ipp, W] 

    #FBA-ODE optimization loop
    ode_data = DataFrame("time" => [0], "fpp" => u0[1], "ipp" => u0[2], "ggp" => u0[3], "phy" => u0[4], "lyc" => u0[5], "bcar" => u0[6], "crtE" => u0[7], "crtB" => u0[8], "crtI" => u0[9], "crtY" => u0[10], "v_p" => [0], "feas" => [1])
    fba_data = DataFrame("time" => [0], "v_in" => [v_in], "v_fpp" => [v_fpp], "v_ipp" => [v_ipp], "lam" => [lam]);

    #println("Beginning loop...")
    for i in 1:N
        #println("Iteration ", i)
        if length(W) == 5
            prob = ODEProblem(beta_carotene_upstream_repression, u0, tspan, p)
        else
            prob = ODEProblem(beta_carotene, u0, tspan, p)
        end
        #Solve ODE
        sol = solve(prob, Rosenbrock23(), reltol=1e-3, abstol=1e-6, saveat=savetimes)

        #Solve for pathway fluxes from final concentrations
        v_p = sol.u[end][7] * michaelismenten_dual(sol.u[end][1], sol.u[end][2], bc_params("kcat_crtE"), bc_params("km_crtE_fpp"), bc_params("km_crtE_ipp"))
    
        #Predict if FBA reaction is feasible using ML model
        flux_data = DataFrame("v_p" => [v_p])
        feas = predict(feas_model, flux_data)
        feas_class = ifelse.(feas .> 0.5, 1, 0)[1]

        if feas_class == 0 && warmup_flag == 1
            break
        end
        
        ode_data = vcat(ode_data, DataFrame("time" => sol.t[1], "fpp" => sol.u[1][1], "ipp" => sol.u[1][2], "ggp" => sol.u[1][3], "phy" => sol.u[1][4], "lyc" => sol.u[1][5], "bcar" => sol.u[1][6], "crtE" => sol.u[1][7], "crtB" => sol.u[1][8], "crtI" => sol.u[1][9], "crtY" => sol.u[1][10], "v_p" => v_p, "feas" => [feas_class]))
        
        flux_data = DataFrame("v_p" => [v_p])
        #Predict new v_in, lam using ML model
        lam = predict(lam_model, flux_data)[1]
        v_in = predict(v_in_model, flux_data)[1]
        v_fpp =  predict(v_fpp_model, flux_data)[1]
        v_ipp = predict(v_ipp_model, flux_data)[1]

        p = [lam, v_in, v_fpp, v_ipp, p[5]] 

        starttime = endtime
        endtime = starttime + deltat
        tspan = [starttime, endtime]
        savetimes = [starttime, endtime]

        u0 = sol.u[end]
        #Save FBA data
        fba_data = vcat(fba_data, DataFrame("time" => [starttime], "v_in" => [p[2]], "v_fpp" => [p[3]], "v_ipp" => [p[4]], "lam" => [p[1]]))
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
function bayesopt_ics(ic_iters, stable_iters, W, fpp_max = 0.755, ipp_max = 0.755, models = [feas_model, lam_model, v_in_model, v_fpp_model, v_ipp_model])
    objectives = []
    fpps = []
    ipps = []
    function ic_objective(params)
        fpp, ipp = values(params)
        u0 = [fpp, ipp, 0., 0., 0., 0., 0., 0., 0., 0.]
        ode_data, fba_data = fba_loop(stable_iters, W, u0, 1, models)
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

function fba_loop_noml(N, W, u0, warmup_flag=0)
    #instantiate initial times
    deltat = 1/(60*60) #genetic timescale, seconds
    starttime = 0.
    endtime = starttime + deltat
    tspan = [starttime, endtime]
    savetimes = [starttime, endtime] #saving start time and end times for alternate v_p calculation

    model = load_model(StandardModel, "C:/Users/Charlotte/OneDrive - University of Edinburgh/Documents/research/joint-simulator/models/iML1515.xml")
    reaction = Reaction("pathway", Dict("M_frdp_c" => -1.0, "M_ipdp_c" => -1.0))
    reaction.lb = 0.0 #Must include lower bound to avoid reaction running backwards with -1000 flux
    add_reactions!(model, [reaction])
    model.reactions["R_EX_glc__D_e"].lb = 0.0
    model.reactions["R_EX_fru_e"].lb = -7.5 #--> results in 0.65 growth rate
    fluxes = flux_balance_analysis_dict(model, Tulip.Optimizer, modifications = [change_constraint("pathway", lb = 0, ub = 0)])
    lam = fluxes["R_BIOMASS_Ec_iML1515_core_75p37M"]
    v_in = - fluxes["R_IPDDI"] + fluxes["R_IPDPS"]
    v_fpp = -fluxes["R_UDCPDPS"] - fluxes["R_HEMEOS"] - fluxes["R_OCTDPS"]
    v_ipp = -8.0*fluxes["R_UDCPDPS"] -5.0*fluxes["R_OCTDPS"] - fluxes["R_DMATT"]

    
    p = [lam, v_in, v_fpp, v_ipp, W] 

    #FBA-ODE optimization loop
    ode_data = DataFrame("time" => [0], "fpp" => u0[1], "ipp" => u0[2], "ggp" => u0[3], "phy" => u0[4], "lyc" => u0[5], "bcar" => u0[6], "crtE" => u0[7], "crtB" => u0[8], "crtI" => u0[9], "crtY" => u0[10], "v_p" => [0], "feas" => [1])
    fba_data = DataFrame("time" => [0], "v_in" => [v_in], "v_fpp" => [v_fpp], "v_ipp" => [v_ipp], "lam" => [lam]);

    #println("Beginning loop...")
    for i in 1:N
        println("Iteration ", i)
        prob = ODEProblem(beta_carotene, u0, tspan, p)
        #Solve ODE
        sol = solve(prob, Rosenbrock23(), reltol=1e-3, abstol=1e-6, saveat=savetimes)

        #Solve for pathway fluxes from final concentrations
        v_p = sol.u[end][7] * michaelismenten_dual(sol.u[end][1], sol.u[end][2], bc_params("kcat_crtE"), bc_params("km_crtE_fpp"), bc_params("km_crtE_ipp"))
    
        #Predict if FBA reaction is feasible using ML model
        fluxes = flux_balance_analysis_dict(model, Tulip.Optimizer, modifications = [change_constraint("pathway", lb = v_p, ub = v_p)])
        if typeof(fluxes) == Nothing
            feas_class = 0
        else feas_class = 1
        end

        if feas_class == 0 && warmup_flag == 1
            break
        end
        
        ode_data = vcat(ode_data, DataFrame("time" => sol.t[1], "fpp" => sol.u[1][1], "ipp" => sol.u[1][2], "ggp" => sol.u[1][3], "phy" => sol.u[1][4], "lyc" => sol.u[1][5], "bcar" => sol.u[1][6], "crtE" => sol.u[1][7], "crtB" => sol.u[1][8], "crtI" => sol.u[1][9], "crtY" => sol.u[1][10], "v_p" => v_p, "feas" => [feas_class]))
        
        #Predict new v_in, lam using ML model
        lam = fluxes["R_BIOMASS_Ec_iML1515_core_75p37M"]
        v_in = - fluxes["R_IPDDI"] + fluxes["R_IPDPS"]
        v_fpp = -fluxes["R_UDCPDPS"] - fluxes["R_HEMEOS"] - fluxes["R_OCTDPS"]
        v_ipp = -8.0*fluxes["R_UDCPDPS"] -5.0*fluxes["R_OCTDPS"] - fluxes["R_DMATT"]


        p = [lam, v_in, v_fpp, v_ipp, p[5]] 

        starttime = endtime
        endtime = starttime + deltat
        tspan = [starttime, endtime]
        savetimes = [starttime, endtime]

        u0 = sol.u[end]
        #Save FBA data
        fba_data = vcat(fba_data, DataFrame("time" => [starttime], "v_in" => [p[2]], "v_fpp" => [p[3]], "v_ipp" => [p[4]], "lam" => [p[1]]))
    end
    return ode_data, fba_data
end