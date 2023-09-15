###ML model training for beta carotene model
###This script generates training data and trains ML models to replace FBA simulation.
using COBREXA
using DataFrames
using Tulip
using CSV
using Plots
using ModelingToolkit
using Statistics
using GLM
using Random
using Flux
using MLBase #Confusion matrix function
using Serialization
println("Completed package import...")

###Generate training data from FBA
function gen_training_data()
    training_data = DataFrame()

    model = load_model(StandardModel, "C:/Users/Charlotte/OneDrive - University of Edinburgh/Documents/research/joint-simulator/models/iML1515.xml")
    reaction = Reaction("pathway", Dict("M_frdp_c" => -1.0, "M_ipdp_c" => -1.0))
    reaction.lb = 0.0 #Must include lower bound to avoid reaction running backwards with -1000 flux
    add_reactions!(model, [reaction])
    model.reactions["R_EX_glc__D_e"].lb = 0.0
    model.reactions["R_EX_fru_e"].lb = -7.5 #--> results in 0.65 growth rate

    for v_p in LinRange(0, 1, 500)
        fluxes = flux_balance_analysis_dict(model, Tulip.Optimizer, modifications = [change_constraint("pathway", lb = v_p, ub = v_p)])
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
    println("Completed training data generation...")
    CSV.write("beta_carotene/ml_models/training_data.csv", training_data)
    return training_data
end

###Train ML models on training data
function train_ml_models(data)
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

    serialize("beta_carotene/ml_models/v_in_model.jls", v_in_model)
    serialize("beta_carotene/ml_models/v_fpp_model.jls", v_fpp_model)
    serialize("beta_carotene/ml_models/v_ipp_model.jls", v_ipp_model)
    serialize("beta_carotene/ml_models/lam_model.jls", lam_model)
    serialize("beta_carotene/ml_models/feas_model.jls", feas_model)
    println("All models saved to JLS files...")
    return v_in_model, v_fpp_model, v_ipp_model, lam_model, feas_model
end

#MAIN
training_data = gen_training_data()
v_in_model, v_fpp_model, v_ipp_model, lam_model, feas_model = train_ml_models(training_data)
println("SCRIPT COMPLETE")