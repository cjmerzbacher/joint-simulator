### BETA CAROTENE MODEL - SIMULATOR AND EXPERIMENTS
#To adapt, change home directory paths
home_path = "/home/cjmerzbacher/joint-simulator/" #for villarica runs
home_path = "C:/Users/Charlotte/OneDrive - University of Edinburgh/Documents/research/joint-simulator/"
drive_path = "F:/"
include(home_path * "models/beta_carotene.jl")

# Import required packages
using DifferentialEquations
using COBREXA
using DataFrames
using Tulip
using Plots
using Colors
using ColorSchemes
using ModelingToolkit
using Statistics
using GLM
using Random
using Flux
using ProgressMeter
using MLBase #Confusion matrix function
using Serialization
using TreeParzen
using CSV
using LatinHypercubeSampling
println("Imports completed")

###GENERATING TRAINING DATA AND ML MODELS 
"""gen_training_data(k::String)
        Generates training data for a given knockout by name. For wild-type, enter an empty string.
""" 
function gen_training_data(k)
    training_data = DataFrame() #Instantiate training data frame

    model = load_model(StandardModel, home_path * "models/iML1515.xml") #Load in FBA model from XML file
    #Modify model to include pathway fluxes
    reaction = Reaction("pathway", Dict("M_frdp_c" => -1.0, "M_ipdp_c" => -1.0)) 
    reaction.lb = 0.0 #Must include lower bound to avoid reaction running backwards with -1000 flux
    add_reactions!(model, [reaction])
    model.reactions["R_EX_glc__D_e"].lb = 0.0
    model.reactions["R_EX_fru_e"].lb = -7.5 #--> results in 0.65 growth rate fit to Borkowski et al. 2018
    
    #Iterate through pathway flux range and simulate FBA
    for v_p in LinRange(0, 1, 500)
        if k == ""
            fluxes = flux_balance_analysis_dict(model, Tulip.Optimizer, modifications = [change_constraint("pathway", lb = v_p, ub = v_p)]) #FBA simulation with pathway flux constraints
        else
            fluxes = flux_balance_analysis_dict(model, Tulip.Optimizer, modifications = [change_constraint("pathway", lb = v_p, ub = v_p), knockout("G_"*k)]) #FBA simulation with pathway flux constraints
        end
            #Check if FBA was feasible
        if typeof(fluxes) != Nothing
            lam = fluxes["R_BIOMASS_Ec_iML1515_core_75p37M"]
            v_in = - fluxes["R_IPDDI"] + fluxes["R_IPDPS"]
            v_fpp = -fluxes["R_UDCPDPS"] - fluxes["R_HEMEOS"] - fluxes["R_OCTDPS"]
            v_ipp = -8.0*fluxes["R_UDCPDPS"] -5.0*fluxes["R_OCTDPS"] - fluxes["R_DMATT"]
            #Save training data
            training_data = vcat(training_data, DataFrame("v_p" => [v_p], "v_in" => [v_in], "lam" => [lam], "v_fpp" => [v_fpp], "v_ipp" => [v_ipp], "feas" => [1]))
        else
            #If infeasible, save training conditions ONLY
            training_data = vcat(training_data, DataFrame("v_p" => [v_p], "v_in" => [NaN], "lam" => [NaN], "v_fpp" => [NaN], "v_ipp" => [NaN], "feas" => [0]))
        end
    end
    println("Completed training data generation for knockout ", k)
    CSV.write(drive_path * "knockouts_ml_models/"*k*"/training_data_"*k*".csv", training_data)
    return training_data
end

"""train_ml_models(knockout::String, data::DataFrame)
        Trains ML surrogate models on a correctly formatted data set for a named knockout.
""" 
function train_ml_models(knockout, data)
    #Train ML models to predict feasibility
    replace!(data.v_in, NaN => -1)
    replace!(data.v_fpp, NaN => -1)
    replace!(data.v_ipp, NaN => -1)
    replace!(data.lam, NaN => -1)

    #Separate data into training and test
    Random.seed!(2023)
    train_indices = randsubseq(1:size(data, 1), 0.8)
    test_indices = setdiff(1:size(data, 1), train_indices)

    train_data = data[train_indices, :]
    test_data = data[test_indices, :]

    #Train a logistic regression model
    println("Classifying flux feasibility...")
    feas_model = glm(@formula(feas~v_p), train_data, Binomial(), LogitLink())

    #Compute training accuracy
    train_pred = predict(feas_model, train_data)
    train_pred_class = ifelse.(train_pred .> 0.5, 1, 0)
    accuracy = sum(train_pred_class .== train_data.feas) / length(train_data.feas)
    println("Accuracy on training set: $accuracy")

    #Generate predictions on test set 
    test_pred = predict(feas_model, test_data)
    test_pred_class = ifelse.(test_pred .> 0.5, 1, 0)
    accuracy = sum(test_pred_class .== test_data.feas) / length(test_data.feas)
    println("Accuracy on test set: $accuracy")

    #Select only feasible data 
    feas_train_indices = [findall(x -> x == 1, train_data.feas)] 
    feas_train_data = train_data[feas_train_indices[1], :]
    feas_test_indices = [findall(x -> x == 1, test_pred_class)] 
    feas_test_data = test_data[feas_test_indices[1], :]

    if nrow(feas_train_data) == 0
        error("No feasible regime found.")
    end

    #Train a linear model to predict v_in
    println("Predicting FPP influx...")
    v_in_model = fit(LinearModel, @formula(v_in~v_p), feas_train_data)

    #Compute training accuracy
    train_pred_in = predict(v_in_model, feas_train_data)
    mse = (mean(feas_train_data.v_in - train_pred_in).^2)
    r2_model = r2(v_in_model)
    println("MSE on training set: $mse")
    println("Model R^2: $r2_model")

    #Train a linear model to predict v_fpp
    println("Predicting FPP influx...")
    v_fpp_model = fit(LinearModel, @formula(v_fpp~v_p), feas_train_data)

    #Compute training accuracy
    train_pred_fpp = predict(v_fpp_model, feas_train_data)
    mse = (mean(feas_train_data.v_fpp - train_pred_fpp).^2)
    r2_model = r2(v_fpp_model)
    println("MSE on training set: $mse")
    println("Model R^2: $r2_model")

    #Train a linear model to predict v_ipp
    println("Predicting IPP influx...")
    v_ipp_model = fit(LinearModel, @formula(v_ipp~v_p), feas_train_data)

    #Compute training accuracy
    train_pred_ipp = predict(v_ipp_model, feas_train_data)
    mse = (mean(feas_train_data.v_ipp - train_pred_ipp).^2)
    r2_model = r2(v_ipp_model)
    println("MSE on training set: $mse")
    println("Model R^2: $r2_model")

    #Train a linear model to predict lam
    println("Predicting growth rate...")
    lam_model = fit(LinearModel, @formula(lam~v_p), feas_train_data)

    #Compute training accuracy
    train_pred_lam = predict(lam_model, feas_train_data)
    mse = (mean(feas_train_data.lam - train_pred_lam).^2)
    r2_model = r2(lam_model)
    println("MSE on training set: $mse")
    println("Model R^2: $r2_model")

    #Compute test accuracy
    test_pred = predict(lam_model, feas_test_data)
    mse = (mean(feas_test_data.lam - test_pred).^2)
    println("MSE on test set: $mse")

    #Serialize model objects
    serialize(drive_path *"knockouts_ml_models/"*knockout*"/v_in_model_"*knockout*".jls", v_in_model)
    serialize(drive_path *"knockouts_ml_models/"*knockout*"/v_fpp_model_"*knockout*".jls", v_fpp_model)
    serialize(drive_path *"knockouts_ml_models/"*knockout*"/v_ipp_model_"*knockout*".jls", v_ipp_model)
    serialize(drive_path *"knockouts_ml_models/"*knockout*"/lam_model_"*knockout*".jls", lam_model)
    serialize(drive_path *"knockouts_ml_models/"*knockout*"/feas_model_"*knockout*".jls", feas_model)
    println("All models saved to JLS files...")
    return v_in_model, v_fpp_model, v_ipp_model, lam_model, feas_model
end

"""read_ml_models(filepath::String, suffix::String)
        Helper function to read in serialized model objects from a filepath with a file suffix (usually knockout).
""" 
function read_ml_models(filepath, suffix)
    if suffix == ""
        v_in_model = deserialize(filepath*"v_in_model.jls")
        v_fpp_model = deserialize(filepath*"v_fpp_model.jls")
        v_ipp_model = deserialize(filepath*"v_ipp_model.jls")
        lam_model = deserialize(filepath*"lam_model.jls")
        feas_model = deserialize(filepath*"feas_model.jls")
    else
        v_in_model = deserialize(filepath*"v_in_model_"*suffix*".jls")
        v_fpp_model = deserialize(filepath*"v_fpp_model_"*suffix*".jls")
        v_ipp_model = deserialize(filepath*"v_ipp_model_"*suffix*".jls")
        lam_model = deserialize(filepath*"lam_model_"*suffix*".jls")
        feas_model = deserialize(filepath*"feas_model_"*suffix*".jls")
    end
    return v_in_model, v_fpp_model, v_ipp_model, lam_model, feas_model
end

###SIMULATOR LOOP
"""fba_loop(N::Integer, W::[Float, Float, Float, Float, Float (optional for upstream repression)], 
                u0::[Float, Float], warmup_flag::Boolean, models::[JLS Object, JLS Object, JLS Object, JLS Object, JLS Object])

        Runs simulator loop for N iterations with promoter strengths W, initial conditions u0 (determined from warmup)
        Models must be valid JLS objects generated from ML training functions or deserialized. 
        Warmup flag determines whether to stop if FBA becomes infeasible.
"""
function fba_loop(N, W, u0, warmup_flag=0, models=[feas_model, lam_model, v_in_model, v_fpp_model, v_ipp_model])
    #Read in surrogate ML models
    feas_model, lam_model, v_in_model, v_fpp_model, v_ipp_model = models
    #instantiate initial times
    deltat = 1/(60*60) #genetic timescale, seconds
    starttime = 0.
    endtime = starttime + deltat
    tspan = [starttime, endtime]
    savetimes = [starttime, endtime] #saving start time and end times for alternate v_p calculation

    #Establish FBA DataFrame first row based on predicted fluxes with noninduced pathway
    fba_df = DataFrame("v_p" => [0.0])
    lam = predict(lam_model, fba_df)[1] #Predict from ML surrogate
    v_in =  predict(v_in_model, fba_df)[1] #Predict from ML surrogate
    v_fpp =  predict(v_fpp_model, fba_df)[1] #Predict from ML surrogate
    v_ipp = predict(v_ipp_model, fba_df)[1] #Predict from ML surrogate
    
    #Create initial parameter vector
    p = [lam, v_in, v_fpp, v_ipp, W] 

    #Save initial time point ODE and FBA data
    ode_data = DataFrame("time" => [0], "fpp" => u0[1], "ipp" => u0[2], "ggp" => u0[3], "phy" => u0[4], "lyc" => u0[5], "bcar" => u0[6], "crtE" => u0[7], "crtB" => u0[8], "crtI" => u0[9], "crtY" => u0[10], "v_p" => [0], "feas" => [1])
    fba_data = DataFrame("time" => [0], "v_in" => [v_in], "v_fpp" => [v_fpp], "v_ipp" => [v_ipp], "lam" => [lam]);
    
    #FBA-ODE optimization loop
    for i in 1:N
        #Determine architecture based on parameter vector length and instantiate correct ODE problem
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

        #If FBA reaction is infeasible and we are not warming up, break the loop and end the simulation
        if feas_class == 0 && warmup_flag == 1
            break
        end
        
        #Save updated ODE data
        ode_data = vcat(ode_data, DataFrame("time" => sol.t[1], "fpp" => sol.u[1][1], "ipp" => sol.u[1][2], "ggp" => sol.u[1][3], "phy" => sol.u[1][4], "lyc" => sol.u[1][5], "bcar" => sol.u[1][6], "crtE" => sol.u[1][7], "crtB" => sol.u[1][8], "crtI" => sol.u[1][9], "crtY" => sol.u[1][10], "v_p" => v_p, "feas" => [feas_class]))
        
        flux_data = DataFrame("v_p" => [v_p])
        #Predict new v_in, lam using ML model
        lam = predict(lam_model, flux_data)[1]
        v_in = predict(v_in_model, flux_data)[1]
        v_fpp =  predict(v_fpp_model, flux_data)[1]
        v_ipp = predict(v_ipp_model, flux_data)[1]

        #Update parameter vector and integration times
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

"""best_in_df(df::DataFrame, col::String)
        Helper function to find lowest objective function in a dataframe column and produce a Boolean mask
"""
function best_in_df(df, col)
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
"""bayesopt_ics(ic_iters::Integer, stable_iters::Integer, W::[Float, Float, Float, Float, Float (optional for upstream repression)])
                fpp_max::Float, ipp_max::Float, models::[JLS Object, JLS Object, JLS Object, JLS Object, JLS Object])
        Bayesian warmup routine to determine initial conditions. Runs for ic_iters and checks if the first stable_iters are feasible, otherwise adds a large penalty to the objective function
        The maximum allowable FPP and IPP concentrations in the search space are fpp_max and ipp_max. 
        ML surrogate models are provided as JLS objects.
"""
function bayesopt_ics(ic_iters, stable_iters, W, fpp_max = 0.755, ipp_max = 0.755, models = [feas_model, lam_model, v_in_model, v_fpp_model, v_ipp_model])
    #Keep track of objective functions and parameter selections
    objectives = []
    fpps = []
    ipps = []

    #NOTE: TreeParzen requires a nested functional structure with an objective function.
    """
        ic_objective(params::[Float, Float])
                Computes scalar objective function by running simulator
    """
    function ic_objective(params)
        fpp, ipp = values(params)
        u0 = [fpp, ipp, 0., 0., 0., 0., 0., 0., 0., 0.]
        #Run test FBA loop for stable_iters iterations
        ode_data, fba_data = fba_loop(stable_iters, W, u0, 1, models)
        #Compute objective function value, adding penalty for infeasibility
        objective = 1/(fpp + ipp)
        if nrow(ode_data) < stable_iters && nrow(fba_data) < stable_iters
            objective = objective + 10E7
        end
        #Save parameter and objective results
        push!(objectives, objective)
        push!(fpps, fpp)
        push!(ipps, ipp)
        return objective
    end

    #Defines Bayesian search space over precursor initial conditions ONLY
    space = Dict(
        :fpp => HP.Uniform(:fpp, 0., fpp_max),
        :ipp => HP.Uniform(:ipp, 0., ipp_max))

    #Run Bayesian optimization with ic_objective
    best = fmin(ic_objective, space, ic_iters)
    fpp_best, ipp_best = values(best)

    #Save data and compute best objective
    data = DataFrame(:fpp => fpps, :ipp => ipps, :ic_obj => objectives)
    data[!, :best_ic] = best_in_df(data, "ic_obj")
    data[!, :w] = fill(W, size(data, 1))

    return fpp_best, ipp_best, minimum(objectives), data
end

###SINGLE FULL SIMULATION
"""single_run(W::[Float, Float, Float, Float, Float (optional for upstream repression)],
                bo_iters::Integer, stable_iters::Integer, sim_iters::Integer, models::[JLS Object, JLS Object, JLS Object, JLS Object, JLS Object])
        Runs a single full simulation for a given promoter strength matrix W. 
        bo_iters is the number of initial condition values to try in the warmup routine
        stable_iters is the number of stable iterations required for a successful warmup routine
        sim_iters is the number of iterations in the final simulation
        ML surrogate models are read in as JLS objects.
"""   
function single_run(W, bo_iters, stable_iters, sim_iters, models)
    #Run Bayesian warmup routine
    fpp_best, ipp_best, objmin, bo_data = bayesopt_ics(bo_iters, stable_iters, W , 10., 10., models)
    #Check if warmup routine failed to converge
    if objmin > 10E4
        println("No feasible ICs found for this promoter strength matrix.")
    end
    println("Initial conditions determined, starting simulation...") 
    #Run simulation with optimal ICs 
    u0 = [fpp_best, ipp_best, 0., 0., 0., 0., 0., 0., 0., 0.]
    ode_data, fba_data = fba_loop(sim_iters, W, u0, 1, models)
    final_lam = fba_data.lam[end]
    min_lam = minimum(fba_data.lam)
    tot_w  = sum(W)
    bcar_tot = sum(ode_data.bcar)
    println("Simulation complete!")

    #Append final simulation data
    fba_data[!, "W"] = fill(W, size(fba_data, 1)) 
    ode_data[!, "W"] = fill(W, size(ode_data, 1)) 

    #Create summary data frame with delta lam, final lam, overall promoter strength, four W values, total beta-carotene production
    summary = DataFrame("w1" => [W[1]], "w2" => [W[2]], "w3" => [W[3]], "w4" => [W[4]], "final_lam" => [final_lam], "delta_lam" => [0.65 - min_lam], "w_tot" => [tot_w], "bcar_tot" => [bcar_tot], "objmin" => [objmin])
    return bo_data, fba_data, ode_data, summary
end

###EXPERIMENT HELPER FUNCTION - BURDEN SWEEP
"""lhc_w_sweep(arch::String, num_iters::Integer, bo_iters::Integer, stable_iters::Integer, save_suffix::String, save_data::Boolean, plan::DataFrame)
        Performs multiple simulations with W values given in plan. 
        Architecture must be specified as a string, either 'ur' or 'nc' for upstream repression vs. no control.
        save_suffix is the folder name (must be created in advance)
        save_data is a flag of whether to save data or not.
        num_iters is the number of W values to run.
"""
function lhc_w_sweep(arch, num_iters, bo_iters, stable_iters, sim_iters, save_suffix, save_data, plan)
    #Instantiate data frames
    global bo_data = DataFrame()
    global sim_fba_data = DataFrame()
    global sim_ode_data = DataFrame()
    global sum_data = DataFrame()
    
    #Iterate through W values 
    for i in 1:num_iters
        print("Beginning iteration ", i)
        if arch == "ur"
            W = values(plan[i, :])[2:6]
        else
            W = values(plan[i, :])[2:5]
        end
        #Run simulation
        bo, fba, ode, sum = single_run(W, bo_iters, stable_iters, sim_iters)
        
        #Save data to DataFrames
        bo_data = vcat(bo_data, bo)
        sim_fba_data = vcat(sim_fba_data, fba)
        sim_ode_data = vcat(sim_ode_data, ode)
        sum_data = vcat(sum_data, sum)

        #Save data to CSV every 5 iterations
        if i%5 == 0 
            if save_data
                #Save out BO data and simulation data
                CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/bo_data_"*save_suffix*".csv", bo_data)
                CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/sim_fba_data_"*save_suffix*".csv", sim_fba_data)
                CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/sim_ode_data_"*save_suffix*".csv", sim_ode_data)
                CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/sum_data_"*save_suffix*".csv", sum_data)
            end
        end
    end
    if save_data
        #Save out BO data and simulation data
        CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/bo_data_"*save_suffix*".csv", bo_data)
        CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/sim_fba_data_"*save_suffix*".csv", sim_fba_data)
        CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/sim_ode_data_"*save_suffix*".csv", sim_ode_data)
        CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/sum_data_"*save_suffix*".csv", sum_data)
    end
    return bo_data, sim_fba_data, sim_ode_data, sum_data
end

###EXPERIMENT HELPER FUNCTION - KNOCKOUTS SCREEN
"""run_knockouts(W::[Float, Float, Float, Float, Float (optional for upstream repression)],
                knockouts::List(String), bo_iters::Integer, sim_iters::Integer, 
                save_data::Boolean, folder_flag::Boolean, model_flag::Boolean, skip_flag::Boolean, sweep_flag::Boolean)
        Runs a list of knockouts. 
        save_data - whether or not data is saved
        folder_flag - whether folders should be created by function
        model_flag - whether models exist already or should be trained
        skip_flag - whether knockouts with folders should be skipped
        sweep_flag - whether to include W values in file names (used if sweeping W)
"""
function run_knockouts(W, knockouts, bo_iters, sim_iters, alt_path, save_data=true, folder_flag=true, model_flag=false, skip_flag=true, sweep_flag=false)
    global bo_data = DataFrame()
    global sim_fba_data = DataFrame()
    global sim_ode_data = DataFrame()
    global sum_data = DataFrame()

    #Iterate through knockouts and run each one
    i = 0
    for k in knockouts
        i = i + 1
        println("Beginning knockout ", i, " of gene ", k)
        if folder_flag==true
            #Check if folder needs to be created
            if isdir(alt_path*"knockouts/"*k)
                #Check if directory already exists
                if skip_flag
                    #Skip if flagged
                    println("Knockout "*k*" already simulated, skipping...")
                    continue
                end
            elseif isdir(alt_path*"knockouts_ml_models/"*k)
                mkdir(alt_path*"knockouts/"*k) #Create necessary directories
                if skip_flag
                    println("Knockout "*k*" models already simulated, skipping...")
                    continue
                end
            else
                #Create necessary dictionaries
                mkdir(alt_path*"knockouts/"*k)
                if model_flag == true 
                    mkdir(alt_path*"knockouts_ml_models/"*k)
                end
            end
        end
        #Check if models already exist
        if model_flag==true
            println("Models already present, reading in...")
            # try
            model_path = "F:/"
            filepath = model_path*"knockouts_ml_models/"*k*"/"
            v_in_model, v_fpp_model, v_ipp_model, lam_model, feas_model = read_ml_models(filepath, k)
            models = [feas_model, lam_model, v_in_model, v_fpp_model, v_ipp_model]
            #Run simulation with ML model passed
            bo_data, sim_fba_data, sim_ode_data, sum_data = single_run(W, bo_iters, 500, sim_iters, models)

            if save_data
                if sweep_flag
                    #Save out BO data and simulation data
                    CSV.write(alt_path * "knockouts/"*k*"/sim_fba_data_"*k*"_"*string(W[1])*".csv", sim_fba_data)
                    CSV.write(alt_path * "knockouts/"*k*"/sim_ode_data_"*k*"_"*string(W[1])*".csv", sim_ode_data)
                    CSV.write(alt_path * "knockouts/"*k*"/sum_data_"*k*"_"*string(W[1])*".csv", sum_data)
                else
                    #Save out BO data and simulation data
                    CSV.write(alt_path * "knockouts/"*k*"/bo_data_"*k*".csv", bo_data)
                    CSV.write(alt_path * "knockouts/"*k*"/sim_fba_data_"*k*".csv", sim_fba_data)
                    CSV.write(alt_path * "knockouts/"*k*"/sim_ode_data_"*k*".csv", sim_ode_data)
                    CSV.write(alt_path * "knockouts/"*k*"/sum_data_"*k*".csv", sum_data)
                end
            end
        else
            #Train new ML model
            training_data = gen_training_data(k)
            try
                v_in_model, v_fpp_model, v_ipp_model, lam_model, feas_model = train_ml_models(k, training_data)
                models = [feas_model, lam_model, v_in_model, v_fpp_model, v_ipp_model]
                #Run simulation with ML model passed
                bo_data, sim_fba_data, sim_ode_data, sum_data = single_run(W, bo_iters, 500, sim_iters, models)

                if save_data
                    #Save out BO data and simulation data
                    CSV.write(alt_path * "knockouts/"*k*"/bo_data_"*k*".csv", bo_data)
                    CSV.write(alt_path * "knockouts/"*k*"/sim_fba_data_"*k*".csv", sim_fba_data)
                    CSV.write(alt_path * "knockouts/"*k*"/sim_ode_data_"*k*".csv", sim_ode_data)
                    CSV.write(alt_path * "knockouts/"*k*"/sum_data_"*k*".csv", sum_data)
                end
            catch
                println("No feasible regime found in training data.")
                continue
            end
        end
    end
end

#EXPERIMENTS
###SINGLE SAMPLE RUN
"""single_run_save()
        Runs and saves out a single run of the simulator for sample data plotting
"""
function single_run_save()
    W = [6.585457605722613e-6, 2.368373589538165e-2, 3.241284507744685e-2, 1.2e-2]
    N = 24*60*60
    u0 = [9.99, 9.99, 0., 0., 0., 0., 0., 0., 0., 0.]
    filepath = "F:/models/beta_carotene/"
    v_in_model, v_fpp_model, v_ipp_model, lam_model, feas_model = read_ml_models(filepath, "")
    ode, fba = fba_loop(N, W, u0, 0, [feas_model, lam_model, v_in_model, v_fpp_model, v_ipp_model])
    CSV.write("F:/plots/sample_bcar_ode_test.csv", ode)
    CSV.write("F:/plots/sample_bcar_fba_test.csv", fba)
end

###KNOCKOUTS SCREEN
"""knockouts_screen()
        Experiment wrapper function to run a list of knockouts for a genome-scale screen
"""
function knockouts_screen()
    W = [2.411031e-07,  0.000097, 0.000098, 0.000367]
    bo_iters = 1000
    sim_iters = 86400
    #List all knockouts to run (this is subset)
    knockouts = ["b1300","b2255","b2486","b3172","b3213","b3876","b4515"]
    run_knockouts(W, knockouts, bo_iters, sim_iters, drive_path, true, false, true)
end

###ADDITIONAL KNOCKOUTS
"""knockouts_experiments()
        Experiment wrapper function to run a list of knockouts and W values
"""
function knockouts_experiments()
    #Select only intermediate knockouts to run from genome-scale screen 
    knockouts = ["b1779", "b2277", "b2779", "b0432", "b0721", "b3919"]
    save_path = "F:/additional_knockouts/"
    bo_iters = 1000
    sim_iters = 86400
    #Select W values (first enzyme only to start?)
    #W_values = [[1.0E-8,  0.000097, 0.000098, 0.000367], [7.5E-9,  0.000097, 0.000098, 0.000367], [5E-9,  0.000097, 0.000098, 0.000367],  [2.5E-9,  0.000097, 0.000098, 0.000367], [10E-8,  0.000097, 0.000098, 0.000367], [7.5E-8,  0.000097, 0.000098, 0.000367], [5E-8,  0.000097, 0.000098, 0.000367], [2.5E-8,  0.000097, 0.000098, 0.000367],  [10E-7,  0.000097, 0.000098, 0.000367], [7.5E-7,  0.000097, 0.000098, 0.000367],  [5E-7,  0.000097, 0.000098, 0.000367]]
    #W_values = [[1.0E-8,  0.000097, 0.000098, 0.000367], [2.5e-7,  0.000097, 0.000098, 0.000367],]
    W_values = [[2.5e-7,  0.000097, 0.000098, 0.000367], [1.0e-7,  0.000097, 0.000098, 0.000367]]
    for W in W_values
        save_path = "F:/additional_knockouts/"
        run_knockouts(W, knockouts, bo_iters, sim_iters, save_path, true, true, true, false, true)
    end
end

###BURDEN
"""burden_experiments()
        Experiment wrapper function to run a list of W values to compare burden and production
"""
function burden_experiments(0)
    num_iters = 400
    bo_iters = 1000
    stable_iters = 500
    sim_iters = 86400

    ## No Control
    scaled_plan = CSV.read(home_path * "data/bcar/lhc.csv", DataFrame)
    save_suffix="nc"
    arch = "nc"
    bo_data, sim_fba_data, sim_ode_data, sum_data = lhc_w_sweep(arch, num_iters, bo_iters, stable_iters, sim_iters, save_suffix, true, plan)

    ## Upstream Repression
    save_suffix="ur"
    arch = "ur"
    ur_scaled_plan = CSV.read(home_path * "beta_carotene/exp_data/lhc_ur.csv", DataFrame)
    bo_data, sim_fba_data, sim_ode_data, sum_data = lhc_w_sweep(arch, num_iters, bo_iters, stable_iters, sim_iters, save_suffix, true, plan)

end

###RUNS ALL EXPERIMENTS - UNCOMMENT DESIRED EXPERIMENTS
# single_run_save()
# knockouts_experiments()
# knockouts_experiments()
# burden_experiments()
println("Experiments completed!")