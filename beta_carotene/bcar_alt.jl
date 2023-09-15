# Import required package
using DifferentialEquations
using COBREXA
using DataFrames
using Tulip
using Plots
using Colors
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
using ColorSchemes

println("Imports completed")

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
            v_fpp = fluxes["R_GRTT"] -fluxes["R_UDCPDPS"] - fluxes["R_HEMEOS"] - fluxes["R_OCTDPS"]
            v_ipp = - fluxes["R_IPDDI"] + fluxes["R_IPDPS"] -8.0*fluxes["R_UDCPDPS"] -5.0*fluxes["R_OCTDPS"] - fluxes["R_DMATT"] - fluxes["R_GRTT"]

            training_data = vcat(training_data, DataFrame("v_p" => [v_p], "lam" => [lam], "v_fpp" => [v_fpp], "v_ipp" => [v_ipp], "feas" => [1]))
        else
            training_data = vcat(training_data, DataFrame("v_p" => [v_p], "lam" => [NaN], "v_fpp" => [NaN], "v_ipp" => [NaN], "feas" => [0]))
        end
    end
    println("Completed training data generation...")
    #CSV.write("beta_carotene/ml_models/training_data.csv", training_data)
    return training_data
end

###Train ML models on training data
function train_ml_models(data)
    #Train ML models to predict feasibility
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
    # serialize("beta_carotene/ml_models/v_fpp_model.jls", v_fpp_model)
    # serialize("beta_carotene/ml_models/v_ipp_model.jls", v_ipp_model)
    # serialize("beta_carotene/ml_models/lam_model.jls", lam_model)
    # serialize("beta_carotene/ml_models/feas_model.jls", feas_model)
    # println("All models saved to JLS files...")
    return v_fpp_model, v_ipp_model, lam_model, feas_model
end

training_data = gen_training_data()
v_fpp_model, v_ipp_model, lam_model, feas_model = train_ml_models(training_data)
println("TRAINING COMPLETE")

michaelismenten(x, kcat, km) = (kcat*x)/(km+x)
michaelismenten_dual(x, y, kcat, km1, km2) = (kcat*((x*y)/(km1*km2)))/(1+(x/km1)+(y/km2))

function bc_params(param_name)
    params = Dict("kcat_crtE" => 0.2456 * 60 * 60, #[1/s] DL,
    "kcat_crtB" => 0.066 * 60 * 60, #[1/s -> 1/hr] DL
    "kcat_crtI" => 4.2255 * 60 * 60, #[1/s] DL
    "kcat_crtY" => 42.9099 * 60 * 60, #[1/s] DL
    "km_crtE_fpp" => 0.0321, #[mM] Brenda
    "km_crtE_ipp" => 0.0234, #[mM] Brenda
    "km_crtB" => 0.01682, #[mM] Brenda
    "km_crtI" => 9.179, #[mM] Brenda
    "km_crtY" => 0.035, #[mM] Brenda
    )
    return params[param_name]
end

function beta_carotene(du, u, p, t)
    lam, v_fpp, v_ipp, W = p
    fpp, ipp, ggp, phy, lyc, bcar, crtE, crtB, crtI, crtY = u
    k_crtE, k_crtB, k_crtI, k_crtY = W
    v_crtE = crtE * michaelismenten_dual(fpp, ipp, bc_params("kcat_crtE"), bc_params("km_crtE_fpp"), bc_params("km_crtE_ipp"))
    v_crtB = crtB * michaelismenten(ggp, bc_params("kcat_crtB"), bc_params("km_crtB"))
    v_crtI = crtI * michaelismenten(phy, bc_params("kcat_crtI"), bc_params("km_crtI"))
    v_crtY = crtY * michaelismenten(lyc, bc_params("kcat_crtY"), bc_params("km_crtY"))

    du[1] = v_fpp - v_crtE - lam*fpp #fpp
    du[2] = v_ipp - v_crtE - lam*ipp #ipp
    du[3] = v_crtE - v_crtB - lam*ggp #ggp
    du[4] = v_crtB - v_crtI - lam*phy #phy
    du[5] = v_crtI - v_crtY - lam*lyc #lyc
    du[6] = v_crtY - lam*bcar #bcar
    du[7] = k_crtE - lam*crtE #crtE
    du[8] = k_crtB - lam*crtB #crtB
    du[9] = k_crtI - lam*crtI #crtI
    du[10] = k_crtY - lam*crtY #crtY
end

function fba_loop(N, W, u0, warmup_flag=0)
    #instantiate initial times
    deltat = 1/(60*60) #genetic timescale, seconds
    starttime = 0.
    endtime = starttime + deltat
    tspan = [starttime, endtime]
    savetimes = [starttime, endtime] #saving start time and end times for alternate v_p calculation

    fba_df = DataFrame("v_p" => [0.0])
    lam = predict(lam_model, fba_df)[1]
    v_fpp =  predict(v_fpp_model, fba_df)[1]
    v_ipp = predict(v_ipp_model, fba_df)[1]
    
    p = [lam, v_fpp, v_ipp, W] 

    #FBA-ODE optimization loop
    ode_data = DataFrame("time" => [0], "fpp" => u0[1], "ipp" => u0[2], "ggp" => u0[3], "phy" => u0[4], "lyc" => u0[5], "bcar" => u0[6], "crtE" => u0[7], "crtB" => u0[8], "crtI" => u0[9], "crtY" => u0[10], "v_p" => [0], "feas" => [1])
    fba_data = DataFrame("time" => [0], "v_fpp" => [v_fpp], "v_ipp" => [v_ipp], "lam" => [lam]);

    #println("Beginning loop...")
    for i in 1:N
        #println("Iteration ", i)
        prob = ODEProblem(beta_carotene, u0, tspan, p)
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
        v_fpp =  predict(v_fpp_model, flux_data)[1]
        v_ipp = predict(v_ipp_model, flux_data)[1]


        p = [lam, v_fpp, v_ipp, p[4]] 

        starttime = endtime
        endtime = starttime + deltat
        tspan = [starttime, endtime]
        savetimes = [starttime, endtime]

        u0 = sol.u[end]
        #Save FBA data
        fba_data = vcat(fba_data, DataFrame("time" => [starttime], "v_fpp" => [p[2]], "v_ipp" => [p[3]], "lam" => [p[1]]))
    end
    return ode_data, fba_data
end

W = [0.00001, 0.0001, 0.001, 0.001]
N = 86400
u0 = [0.7, 0.7, 0., 0., 0., 0., 0., 0., 0., 0.]

ode_data, fba_data = fba_loop(N, W, u0, 1)

palette= ColorSchemes.tab10.colors
p1 = plot(fba_data.time, fba_data.lam, lw=3, legend=false, color="black", xlabel="time (hrs)", ylabel="Growth rate (mM/hr)")
p2 = plot(fba_data.time, [fba_data.v_fpp fba_data.v_ipp],  label=["FPP" "IPP"], color=[palette[3] palette[4]], lw=3,xlabel="time (hrs)", ylabel="Flux (mM/hr)")
p3 = plot(ode_data.time, ode_data.v_p,lw=3, label="Pathway", color=palette[2], xlabel="time (hrs)", ylabel="Flux (mM/hr)")
p4 = plot(ode_data.time, [ode_data.crtE ode_data.crtB ode_data.crtI ode_data.crtY],lw=3,label=["CrtE" "CrtB" "CrtI" "CrtY"], color=[palette[2] palette[3] palette[4] palette[5]], xlabel="time (hrs)", ylabel="Concentration (mM)")
p5 = plot(ode_data.time, [ode_data.fpp ode_data.ipp], lw=3, label=["FPP" "IPP"], color=[palette[5] palette[9]], xlabel="time (hrs)", ylabel="Concentration (mM)")
p6 = plot(ode_data.time, [ode_data.ggp ode_data.phy ode_data.lyc ode_data.bcar], lw=3, label=["GGP" "Phy" "Lycopene" "Beta-carotene"], color=[palette[7] palette[8] palette[10] palette[1]], xlabel="time (hrs)", ylabel="Concentration (mM)")

plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(700, 700))