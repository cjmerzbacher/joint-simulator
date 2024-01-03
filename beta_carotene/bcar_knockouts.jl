# Import required package
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
#home_path = "/home/cjmerzbacher/joint-simulator/" #for villarica runs
home_path = "C:/Users/Charlotte/OneDrive - University of Edinburgh/Documents/research/joint-simulator/"
alt_path="F:/"
include(home_path * "models/beta_carotene.jl")
include(home_path * "beta_carotene/bcar_sim.jl")

println("Imports completed")

### Run a single simulation, selecting appropriate ICs
function single_run(W, bo_iters, stable_iters, sim_iters, models)
    fpp_best, ipp_best, objmin, bo_data = bayesopt_ics(bo_iters, stable_iters, W , 10., 10., models)
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

###Generate training data from FBA
function gen_training_data(k)
    training_data = DataFrame()

    model = load_model(StandardModel, home_path * "models/iML1515.xml")
    reaction = Reaction("pathway", Dict("M_frdp_c" => -1.0, "M_ipdp_c" => -1.0))
    reaction.lb = 0.0 #Must include lower bound to avoid reaction running backwards with -1000 flux
    add_reactions!(model, [reaction])
    model.reactions["R_EX_glc__D_e"].lb = 0.0
    model.reactions["R_EX_fru_e"].lb = -7.5 #--> results in 0.65 growth rate
    

    for v_p in LinRange(0, 1, 500)
        fluxes = flux_balance_analysis_dict(model, Tulip.Optimizer, modifications = [change_constraint("pathway", lb = v_p, ub = v_p), knockout("G_"*k)])
        if typeof(fluxes) != Nothing
            lam = fluxes["R_BIOMASS_Ec_iML1515_core_75p37M"]
            v_in = - fluxes["R_IPDDI"] + fluxes["R_IPDPS"]
            v_fpp = -fluxes["R_UDCPDPS"] - fluxes["R_HEMEOS"] - fluxes["R_OCTDPS"]
            v_ipp = -8.0*fluxes["R_UDCPDPS"] -5.0*fluxes["R_OCTDPS"] - fluxes["R_DMATT"]

            training_data = vcat(training_data, DataFrame("v_p" => [v_p], "v_in" => [v_in], "lam" => [lam], "v_fpp" => [v_fpp], "v_ipp" => [v_ipp], "feas" => [1]))
        else
            training_data = vcat(training_data, DataFrame("v_p" => [v_p], "v_in" => [NaN], "lam" => [NaN], "v_fpp" => [NaN], "v_ipp" => [NaN], "feas" => [0]))
        end
    end
    println("Completed training data generation for knockout ", k)
    CSV.write(alt_path * "knockouts_ml_models/"*k*"/training_data_"*k*".csv", training_data)
    return training_data
end

###Train ML models on training data
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

    serialize(alt_path *"knockouts_ml_models/"*knockout*"/v_in_model_"*knockout*".jls", v_in_model)
    serialize(alt_path *"knockouts_ml_models/"*knockout*"/v_fpp_model_"*knockout*".jls", v_fpp_model)
    serialize(alt_path *"knockouts_ml_models/"*knockout*"/v_ipp_model_"*knockout*".jls", v_ipp_model)
    serialize(alt_path *"knockouts_ml_models/"*knockout*"/lam_model_"*knockout*".jls", lam_model)
    serialize(alt_path *"knockouts_ml_models/"*knockout*"/feas_model_"*knockout*".jls", feas_model)
    println("All models saved to JLS files...")
    return v_in_model, v_fpp_model, v_ipp_model, lam_model, feas_model
end


function run_knockouts(W, knockouts, bo_iters, sim_iters, save_data=true)
    global bo_data = DataFrame()
    global sim_fba_data = DataFrame()
    global sim_ode_data = DataFrame()
    global sum_data = DataFrame()

    i = 0
    for k in knockouts
        i = i + 1
        println("Beginning knockout ", i, " of gene ", k)
        # if isdir(alt_path*"knockouts/"*k)
        #     println("Knockout "*k*" already simulated, skipping...")
        # elseif isdir(alt_path*"knockouts_ml_models/"*k)
        #     println("Knockout "*k*" already simulated, skipping...")
        # else
            #Create necessary dictionaries
            # mkdir(alt_path*"knockouts/"*k)
            # mkdir(alt_path*"knockouts_ml_models/"*k)
            
        #TRAIN NEW ML model 
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

           
        # end
    end
end

W = [2.411031e-07,  0.000097, 0.000098, 0.000367]
bo_iters = 1000
sim_iters = 86400
#knockouts = ["b0573", "b2957", "b0150", "b2253", "b2254", "b1967", "b2256", "b0933", "b1865", "b2258", "b0934", "b4544", "b0936", "b2954", "b3621", "b2032", "b1781", "b2027", "b1002", "b3001", "b2938", "b4090", "b2035", "b3620", "b3622", "b3624", "b0459", "b4084", "b3629", "b4085", "b0585", "b3628", "b3748", "b3627", "b3751", "b1326", "b3850", "b1325", "b3750", "b3630", "b4230", "b0591", "b3632", "b3035", "b4227", "b3625", "b4231", "b0692", "b2810", "b3749", "b4485", "b2014", "b0260", "b1680", "b1296", "b3867", "b1298", "b3374", "b0810", "b0809", "b3824", "b1308", "b2133", "b0811", "b2260", "b1492", "b3540", "b1487", "b3544", "b0166", "b2893", "b1486", "b1483", "b0604", "b1484", "b3930", "b1519", "b1485", "b1511", "b2371", "b2989", "b2302", "b1619", "b0785", "b3893", "b1474", "b3894", "b1475", "b3892", "b1476", "b0776", "b3412", "b2515", "b4238", "b4237", "b3924", "b4209", "b1387", "b1393", "b2912"]
knockouts = ["b1993","b2724","b2935","b3034","b3041","b1241","b2021","b2232","b2418","b2481","b3005","b3784","b3831","b4481"]
run_knockouts(W, knockouts, bo_iters, sim_iters, true)

# knockouts = CSV.read(home_path * "data/bcar/knockouts/remaining_knockouts.csv", DataFrame)
# run_knockouts(W, knockouts.knockouts[103:end], bo_iters, sim_iters, true)

# feas_model = deserialize(home_path * "beta_carotene/ml_models/knockouts/b2551/feas_model_b2551.jls")
# lam_model = deserialize(home_path * "beta_carotene/ml_models/knockouts/b2551/lam_model_b2551.jls")
# v_in_model = deserialize(home_path * "beta_carotene/ml_models/knockouts/b2551/v_in_model_b2551.jls")
# v_fpp_model = deserialize(home_path * "beta_carotene/ml_models/knockouts/b2551/v_fpp_model_b2551.jls")
# v_ipp_model = deserialize(home_path * "beta_carotene/ml_models/knockouts/b2551/v_ipp_model_b2551.jls")
# models =  [feas_model, lam_model, v_in_model, v_fpp_model, v_ipp_model]
# bo, fba, ode, summary = single_run(W, bo_iters, 500, sim_iters, models)
# W = [0.00001, 0.0001, 0.001, 0.001]
# N = 100
# u0 = [0.7, 0.7, 0., 0., 0., 0., 0., 0., 0., 0.]

#ode_data, fba_data = fba_loop_noml(N, W, u0, 1)
# palette= ColorSchemes.tab10.colors
# p1 = plot(fba_data.time, fba_data.lam, lw=3, legend=false, color="black", xlabel="time (hrs)", ylabel="Growth rate (mM/hr)")
# p2 = plot(fba_data.time, [fba_data.v_in fba_data.v_fpp fba_data.v_ipp],  label=["Influx" "FPP" "IPP"], color=[palette[1] palette[3] palette[4]], lw=3,xlabel="time (hrs)", ylabel="Flux (mM/hr)")
# p3 = plot(ode_data.time, ode_data.v_p,lw=3, label="Pathway", color=palette[2], xlabel="time (hrs)", ylabel="Flux (mM/hr)")
# p4 = plot(ode_data.time, [ode_data.crtE ode_data.crtB ode_data.crtI ode_data.crtY],lw=3,label=["CrtE" "CrtB" "CrtI" "CrtY"], color=[palette[2] palette[3] palette[4] palette[5]], xlabel="time (hrs)", ylabel="Concentration (mM)")
# p5 = plot(ode_data.time, [ode_data.fpp ode_data.ipp], lw=3, label=["FPP" "IPP"], color=[palette[5] palette[9]], xlabel="time (hrs)", ylabel="Concentration (mM)")
# p6 = plot(ode_data.time, [ode_data.ggp ode_data.phy ode_data.lyc ode_data.bcar], lw=3, label=["GGP" "Phy" "Lycopene" "Beta-carotene"], color=[palette[7] palette[8] palette[10] palette[1]], xlabel="time (hrs)", ylabel="Concentration (mM)")

# plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(700, 700))