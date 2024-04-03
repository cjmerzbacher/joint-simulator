### BETA CAROTENE MODEL - SIMULATOR AND EXPERIMENTS
#To adapt, change home directory paths
home_path = "/home/cjmerzbacher/joint-simulator/" #for villarica runs
home_path = "C:/Users/Charlotte/OneDrive - University of Edinburgh/Documents/research/joint-simulator/"

include(home_path * "models/glucaric_acid.jl")

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
println("Imports completed")

###GENERATING TRAINING DATA AND ML MODELS 
"""gen_training_data(cond_name::String)
        Generates training data for a given medium condition by name.
"""  
function gen_training_data(cond_name)
    training_data = DataFrame() #Instantiate training data frame

    model = load_model(StandardModel, home_path * "models/iML1515.xml") #Load in FBA model from XML file
    #Modify model to include pathway fluxes
    reaction = Reaction("pathway", Dict("M_g6p_c" => -1.0))
    reaction.lb = 0.0 #Must include lower bound to avoid reaction running backwards with -1000 flux
    add_reactions!(model, [reaction])
    medium = model.medium
    medium["R_EX_glc__D_e"] = 0.0
    medium[gc] = 10.0
    model.medium = medium

    #Iterate through pathway flux range and simulate FBA
    for v_p in LinRange(0, 1, 500)
        #FBA simulation with pathway flux constraints
        fluxes = flux_balance_analysis_dict(model, Tulip.Optimizer, modifications = [change_constraint("pathway", lb = v_p, ub = v_p)])
        #Check if FBA was feasible
        if typeof(fluxes) != Nothing
            lam = fluxes["R_BIOMASS_Ec_iML1515_core_75p37M"]
            v_in = fluxes["R_TRE6PH"] + fluxes["R_PGMT"] + fluxes["R_HEX1"] + fluxes["R_AB6PGH"] + fluxes["R_TRE6PS"] + fluxes["R_FRULYSDG"] + fluxes["R_GLCptspp"] + fluxes["R_G6PP"] + fluxes["R_G6Pt6_2pp"] + fluxes["R_BGLA1"]
            #Save training data
            training_data = vcat(training_data, DataFrame("v_p" => [v_p], "v_in" => [v_in], "lam" => [lam], "feas" => [1]))
        else
            #If infeasible, save training conditions ONLY
            training_data = vcat(training_data, DataFrame("v_p" => [v_p], "v_in" => [NaN], "lam" => [NaN], "feas" => [0]))
        end
    end
    println("Completed training data generation...")
    CSV.write(home_path * "data/ga/medium_conditions/training_data"*cond_name*".csv", training_data)
    return training_data
end

"""train_ml_models(data::DataFrame)
        Trains ML surrogate models on a correctly formatted data set.
""" 
function train_ml_models(data)
    #Train ML models to predict feasibility
    replace!(data.v_in, NaN => -1)
    replace!(data.lam, NaN => -1)

    #Separate data into training and test
    Random.seed!(2023)
    train_indices = randsubseq(1:size(data, 1), 0.8)
    test_indices = setdiff(1:size(data, 1), train_indices)

    train_data = data[train_indices, :]
    test_data = data[test_indices, :]

    #Train a logistic regression model
    println("Classifying flux feasibility...")
    feas_model = glm(@formula(feas~v_dp), train_data, Binomial(), LogitLink())

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

    #Create and plot confusion matrix
    cm = confusmat(2, train_data.feas.+ 1, train_pred_class.+ 1)
    p1 = heatmap(cm, c=:blues, xlabel="Predicted", ylabel="True", title="Training Set", xticks=nothing, yticks=nothing, colorbar=false, legend=nothing)
    annotate!([(i, j, text(cm[i, j])) for i in 1:size(cm)[1] for j in 1:size(cm)[2]])

    cm = confusmat(2, test_data.feas.+ 1, test_pred_class.+ 1)
    p2 = heatmap(cm, c=:blues, xlabel="Predicted", ylabel="True", title="Test Set", xticks=nothing, yticks=nothing, colorbar=false, legend=nothing)
    annotate!([(i, j, text(cm[i, j])) for i in 1:size(cm)[1] for j in 1:size(cm)[2]])

    plot(p1, p2, layout = (1, 2), dpi=300, size=(400, 220))

    #Select only feasible data 
    feas_train_indices = [findall(x -> x == 1, train_data.feas)] 
    feas_train_data = train_data[feas_train_indices[1], :]
    feas_test_indices = [findall(x -> x == 1, test_pred_class)] 
    feas_test_data = test_data[feas_test_indices[1], :]
    
    #Train a neural network model to predict v_in
    println("Predicting pathway influx...")
    num_epochs = 1000
    step_size = 0.01
    num_units = 500
    #Create model architecture
    v_in_model = Chain(Dense(1, num_units, relu), Dense(num_units, num_units, relu), Dense(num_units, 1))

    #Transpose train and test data from DataFrame
    x_train = feas_train_data.v_dp'
    y_train = feas_train_data.v_in'
    x_test = feas_test_data.v_dp'
    y_test = feas_test_data.v_in'

    #Define loss function and optimizer
    loss(v_in_model, x, y) = mean(abs2.(v_in_model(x) .- y))
    opt = Flux.setup(Adam(step_size), v_in_model)

    #training loop
    losses = []
    for epoch in 1:num_epochs
        Flux.train!(loss, v_in_model, [(x_train, y_train)], opt)
        push!(losses, loss(v_in_model, x_train, y_train))
    end

    #Compute training loss
    train_loss = losses[end]
    println("Final training loss: $train_loss")

    #Compute test loss
    test_loss = loss(v_in_model, x_test, y_test)
    println("Test loss: $test_loss")

    #Train a linear model to predict lam
    println("Predicting growth rate...")
    lam_model = fit(LinearModel, @formula(lam~v_dp), feas_train_data)

    #Compute training accuracy
    train_pred = predict(lam_model, feas_train_data)
    mse = (mean(feas_train_data.lam - train_pred).^2)
    r2_model = r2(lam_model)
    println("MSE on training set: $mse")
    println("Model R^2: $r2_model")

    #Compute test accuracy
    test_pred = predict(lam_model, feas_test_data)
    mse = (mean(feas_test_data.lam - test_pred).^2)
    println("MSE on test set: $mse")
    return feas_model, v_in_model, lam_model
end

###SIMULATOR LOOP
"""fba_loop(N::Integer, A_W::[Architecture, W_matrix], 
                u0::[Float], warmup_flag::Boolean)

        Runs simulator loop for N iterations with promoter strengths W and architecture, initial conditions u0 (determined from warmup)
        Warmup flag determines whether to stop if FBA becomes infeasible.
"""
function fba_loop(N, A_W, u0, warmup_flag=0, models=[v_in_model, lam_model, feas_model])
    v_in_model, lam_model, feas_model = models
    #instantiate initial times
    deltat = 1/(60*60) #genetic timescale, seconds
    starttime = 0.
    endtime = starttime + deltat
    tspan = [starttime, endtime]
    savetimes = [starttime, endtime] #saving start time and end times for alternate v_dp calculation

    fba_df = DataFrame("v_dp" => [0.0])
    lam = predict(lam_model, fba_df)[1]
    v_in = v_in_model(fba_df.v_dp')[1]
    A, W = A_W
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
"""warmup(num_iters::Integer, A_W::[Architectre, W_matrix]), f6p_max::Float, g6p_max::Float)
        Bayesian warmup routine to determine initial conditions. Runs for num_iters and checks if the first 500 are feasible, otherwise adds a large penalty to the objective function
        The maximum allowable f6p and g6p concentrations in the search space are f6p_max and g6p_max. 
        ML surrogate models are provided as JLS objects.
"""
function warmup(num_iters, A_W, f6p_max = 0.281, g6p_max = 0.068)
    objectives = []
    f6ps = []
    g6ps = []

    function warmup_objective(params)
        f6p, g6p = values(params)
        u0 = [g6p, f6p, 0., 0., 0.]
        stable_iters = 500
        ode_data, fba_data = fba_loop(stable_iters, A_W, u0, 1)
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

###SINGLE FULL SIMULATION
"""single_run(params::[Architecture, W_matrix], bo_iters::Integer, sim_iters::Integer)
        Runs a single full simulation for a given promoter strength matrix W and a genetic control architecture. 
        bo_iters is the number of initial condition values to try in the warmup routine
        sim_iters is the number of iterations in the final simulation
"""   
function single_run(params, bo_iters, sim_iters)
    A, W = params

    f6p_best, g6p_best, objmin, bo_data = warmup(bo_iters, params) 
    if objmin > 10E4
        println("No feasible ICs found for this promoter strength matrix.")
    end
    println("Initial conditions determined, starting simulation...") 
    #Run simulation with optimal ICs 
    u0 = [f6p_best, g6p_best, 0., 0., 0.]
    ode_data, fba_data = fba_loop(sim_iters, params, u0, 1)
    println("Simulation complete!")

    #Append final simulation data
    fba_data[!, "W"] = fill(W, size(fba_data, 1)) 
    ode_data[!, "W"] = fill(W, size(ode_data, 1)) 
    fba_data[!, "k_ino1"] = fill(W[1][3], size(fba_data, 1)) 
    ode_data[!, "k_ino1"] = fill(W[1][3], size(ode_data, 1)) 

    delta_burden = sum(fba_data.lam[1] - fba_data.lam[end])
    burden = (1-(fba_data.lam[1]-delta_burden)/fba_data.lam[1])
    ga_ss = ode_data.mi[end]
    production = sum((ga_ss .- ode_data.mi).^2)

    #Create summary data frame with delta lam, final lam, overall promoter strength, four W values, total beta-carotene production
    summary = DataFrame("k_ino1" => [W[1][3]], "theta_ino1" => [W[1][2]], "k_miox" => [W[2][3]], "theta_miox" => [W[2][2]], "production" => [production], "burden" => [burden], "objmin" => [objmin])
    
    return fba_data, ode_data, summary
end

###BAYESOPT LOOP
function bayesopt(alpha, num_iters, bo_iters, sim_iters, save_suffix, save_data=true)
    global sim_fba_data = DataFrame()
    global sim_ode_data = DataFrame()
    global sum_data = DataFrame()
    global bo_data = DataFrame()

    global i = 0
    global a = alpha
    global s = 1256

    #Define objective function
    function objective(args)
        i = i+1
        println("Beginning iteration ", i)
        A = args[:architecture]
        k_ino1 = args[:k_ino1]
        theta_ino1 = args[:theta_ino1]
        k_miox = args[:k_miox]
        theta_miox = args[:theta_miox]
        W = [[2., theta_ino1, k_ino1], [2., theta_miox, k_miox]]
        params = [A, W]

        fba, ode, summary = single_run(params, bo_iters, sim_iters)

        bo = DataFrame("arch" => [A], "k_ino1" => [k_ino1], "theta_ino1" => [theta_ino1], "k_miox" => [k_miox] , "theta_miox" => [theta_miox], "objective" => [summary.burden[1]/summary.production[1]], "burden" => [summary.burden[1]], "production" => [summary.production[1]])

        obj = a*summary.burden[1] + ((1-a)*s)/summary.production[1]
        println("Objective is ", obj, " with a burden of ", summary.burden[1], " and a production of ", summary.production[1])
        summary[!, "obj"] = fill(obj, size(summary, 1))

        sim_fba_data = vcat(sim_fba_data, fba)
        sim_ode_data = vcat(sim_ode_data, ode)
        sum_data = vcat(sum_data, summary)
        bo_data = vcat(bo_data, bo)

        if i%5 == 0 
            if save_data
                #Save out simulation data
                CSV.write("F:/bayesopt/"*save_suffix*"/sim_fba_data_"*save_suffix*".csv", sim_fba_data)
                CSV.write("F:/bayesopt/"*save_suffix*"/sim_ode_data_"*save_suffix*".csv", sim_ode_data)
                CSV.write("F:/bayesopt/"*save_suffix*"/sum_data_"*save_suffix*".csv", sum_data)
                CSV.write("F:/bayesopt/"*save_suffix*"/bo_data_"*save_suffix*".csv", bo_data)
            end
        end

        
        #Compute objective function and save out
        return obj
    end

    #Establish search space
    space = Dict(
        :architecture => HP.Choice(:architecture, [[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]]]),
        :k_ino1 => HP.Uniform(:k_ino1, 10E-7, 10E-3),
        :theta_ino1 => HP.Uniform(:theta_ino1, 10E-7, 10.),
        :k_miox => HP.Uniform(:k_miox, 10E-7, 10E-3),
        :theta_miox => HP.Uniform(:theta_miox, 10E-7, 10.))
    
    #Run bayesopt
    best = fmin(objective, space, num_iters)

    if save_data
        #Save out simulation data
        CSV.write("F:/bayesopt/"*save_suffix*"/sim_fba_data_"*save_suffix*".csv", sim_fba_data)
        CSV.write("F:/bayesopt/"*save_suffix*"/sim_ode_data_"*save_suffix*".csv", sim_ode_data)
        CSV.write("F:/bayesopt/"*save_suffix*"/sum_data_"*save_suffix*".csv", sum_data)
        CSV.write("F:/bayesopt/"*save_suffix*"/bo_data_"*save_suffix*".csv", bo_data)
    end
end

###BURDEN RUN
function burden(A, Ws, num_iters, bo_iters, sim_iters, save_suffix, save_data=true)
    global bo_data = DataFrame()
    global sim_fba_data = DataFrame()
    global sim_ode_data = DataFrame()
    global sum_data = DataFrame()

    for i in 1:num_iters
        print("Beginning iteration ", i)
        index, k_ino1, theta_ino1, k_miox, theta_miox = values(Ws[i, :])
        W = [[2., theta_ino1, k_ino1], [2., theta_miox, k_miox]]
        params = [A, W]
        bo, fba, ode, summary = single_run(params, bo_iters, sim_iters)

        bo_data = vcat(bo_data, bo)
        sim_fba_data = vcat(sim_fba_data, fba)
        sim_ode_data = vcat(sim_ode_data, ode)
        sum_data = vcat(sum_data, summary)

        if i%5 == 0 
            if save_data
                #Save out BO data and simulation data
                CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/bo_data_"*save_suffix*".csv", bo_data)
                CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sim_fba_data_"*save_suffix*".csv", sim_fba_data)
                CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sim_ode_data_"*save_suffix*".csv", sim_ode_data)
                CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sum_data_"*save_suffix*".csv", sum_data)
            end
        end
    end
    if save_data
        #Save out BO data and simulation data
        CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/bo_data_"*save_suffix*".csv", bo_data)
        CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sim_fba_data_"*save_suffix*".csv", sim_fba_data)
        CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sim_ode_data_"*save_suffix*".csv", sim_ode_data)
        CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sum_data_"*save_suffix*".csv", sum_data)
    end
    return bo_data, sim_fba_data, sim_ode_data, sum_data
end

#EXPERIMENTS
###SINGLE SAMPLE RUN
"""single_run_save()
        Runs and saves out a single run of the simulator for sample data plotting
"""
function single_run_save(N)
    #A = [[0, 0, 1], [0, 0, 1]] #open loop
    # W = [[2., 3.328086, 4.1e-6], [2., 5.070964, 2.2e-5]]
    A = [[0, 1, 0], [1, 0, 0]] #dual control 
    W = [[2., 3.328086, 2e-5], [2., 1, 2.2e-3]]
    A_W = A, W
    u0 = [0.067, 0.280, 0., 0., 0.]
    filepath = "F:/models/glucaric_acid/"
    v_in_model = deserialize(filepath*"v_in_model.jls")
    lam_model = deserialize(filepath*"lam_model.jls")
    feas_model = deserialize(filepath*"feas_model.jls")
    models = [v_in_model, lam_model, feas_model]
    ode, fba = fba_loop(N, A_W, u0, 0, models)
    # CSV.write("F:/plots/sample_ga_ode_test.csv", ode)
    # CSV.write("F:/plots/sample_ga_fba_test.csv", fba)
end

# ###MEDIUM CONDITIONS  
# function medium_conditions_experiments()
#     #Read in and train ML models
#     growth_conditions_names = ["galactose", "gluconate", "xylose", "lalanine", "lactate", "pyruvate", "ribose", "glucose", "fructose", "sorbitol", "mannitol", "na-glucosamine", "glycerol", "succinate", "acetate"]

#     p_aucs = []
#     p_finals = []
#     for g in growth_conditions_names
#         println("Simulating growth conditions ", g)
#         try
#             data = CSV.read("F:/medium_conditions/training_"*g*".csv", DataFrame) 

#             feas_model, v_in_model, lam_model = train_ml_models(data)

#             println(g)
#             fba_df = DataFrame("v_dp" => [0.0])
#             lam = predict(lam_model, fba_df)[1]
#             println(lam)
#             if lam_model == 0
#                 print("No feasible FBA areas")
#                 continue
#             end

#             #Specify algorithm parameters
#             A = [[0, 0, 1], [0, 0, 1]] #open loop
#             W = [[2., 3.328086, 0.000041], [2., 5.070964, 0.000227]] #Optimal from MSc work - glucaric_acid_singlearch.csv

#             #Use ML model to predict initial lam, v_in
#             fba_df = DataFrame("v_dp" => [0.0])
#             lam = predict(lam_model, fba_df)[1]
#             v_in = v_in_model(fba_df.v_dp')[1]
#             global p
#             p = [A, W, v_in, lam, feas_model, v_in_model, lam_model]

#             ho = @hyperopt for i=10,
#                 sampler = RandomSampler(), # This is default if none provided
#                 g6p = LinRange(0., 0.281, 1000),
#                 f6p = LinRange(0.,0.068,1000)
#                 warmup_objective(g6p, f6p)
#             end

#             initial_conditions, objective_min = ho.minimizer, ho.minimum

#             N = 86400
#             u0 = [initial_conditions[1], initial_conditions[2], 0., 0., 0.]
#             ode_data, fba_data = fba_loop(N, p, u0, 0)
#             push!(p_aucs, sum(ode_data.mi))
#             push!(p_finals, last(ode_data.mi))

#             CSV.write(home_path * "data/ga/medium_conditions/"*g*"_ode_data.csv", ode_data) #FIX THIS
#             CSV.write(home_path * "data/ga/medium_conditions/"*g*"_fba_data.csv", fba_data) #FIX THIS
#         catch 
#             println("Data file not written for condition "*g)
#         end
#     end
# end

###BURDEN EXPERIMENTS
function burden_experiments()
    println("OPEN LOOP CONTROL")
    A = [[0, 0, 1], [0, 0, 1]] #open loop
    arch = "nc"
    save_suffix = arch
    num_iters = 400
    bo_iters = 1000
    sim_iters = 86400

    scaled_plan = CSV.read(home_path * "data/ga/"*arch*"_lhc.csv", DataFrame)
    bo_data, sim_fba_data, sim_ode_data, sum_data = burden(A, scaled_plan, num_iters, bo_iters, sim_iters, save_suffix, true)

    println("DUAL CONTROL")
    A = [[0, 0, 1], [0, 0, 1]] #open loop
    arch = "dc"
    save_suffix=arch
    num_iters = 400
    bo_iters = 1000
    sim_iters = 86400

    scaled_plan = CSV.read(home_path * "data/ga/burden/"*arch*"_lhc.csv", DataFrame)
    bo_data, sim_fba_data, sim_ode_data, sum_data = new_name(A, scaled_plan, num_iters, bo_iters, sim_iters, save_suffix, true)

    println("UPSTREAM REPRESSION")
    A = [[0, 1, 0], [0, 0, 1]] 
    arch = "ur"
    save_suffix=arch
    num_iters = 400
    bo_iters = 1000
    sim_iters = 86400

    scaled_plan = CSV.read(home_path * "data/ga/burden/"*arch*"_lhc.csv", DataFrame)
    bo_data, sim_fba_data, sim_ode_data, sum_data = new_name(A, scaled_plan, num_iters, bo_iters, sim_iters, save_suffix, true)

    println("DOWNSTREAM ACTIVATION")
    A = [[0, 0, 1], [1, 0, 0]] 
    arch = "da"
    save_suffix=arch
    num_iters = 400
    bo_iters = 1000
    sim_iters = 86400

    scaled_plan = CSV.read(home_path * "data/ga/burden/"*arch*"_lhc.csv", DataFrame)
    bo_data, sim_fba_data, sim_ode_data, sum_data = new_name(A, scaled_plan, num_iters, bo_iters, sim_iters, save_suffix, true)

end

###BAYESOPT EXPERIMENTS
function bayesopt_experiments()
    #Read in saved models
    feas_model = deserialize(home_path * "models/ml_models/ga/feas_model.jls")
    lam_model = deserialize(home_path * "models/ml_models/ga/lam_model.jls")
    v_in_model = deserialize(home_path * "models/ml_models/ga/v_in_model.jls")
    println("All models read in successfully!")

    num_iters = 100
    bo_iters = 1000
    sim_iters = 86400
    bayesopt(0.0, num_iters, bo_iters, sim_iters, "bayesopt_0_2", true)
    # bayesopt(0.01, num_iters, bo_iters, sim_iters, "bayesopt_01", true)
    # bayesopt(0.02, num_iters, bo_iters, sim_iters, "bayesopt_02", true)
    # bayesopt(0.03, num_iters, bo_iters, sim_iters, "bayesopt_03", true)
    # bayesopt(0.04, num_iters, bo_iters, sim_iters, "bayesopt_04", true)
    # bayesopt(0.05, num_iters, bo_iters, sim_iters, "bayesopt_05", true)
    # bayesopt(0.1, num_iters, bo_iters, sim_iters, "bayesopt_10", true)
    # bayesopt(0.15, num_iters, bo_iters, sim_iters, "bayesopt_15", true)
    # bayesopt(0.2, num_iters, bo_iters, sim_iters, "bayesopt_20", true)
    # bayesopt(0.25, num_iters, bo_iters, sim_iters, "bayesopt_25", true)
    bayesopt(0.5, num_iters, bo_iters, sim_iters, models, "bayesopt_5_2", true)
    # bayesopt(0.75, num_iters, bo_iters, sim_iters, "bayesopt_75", true)
    bayesopt(1.0, num_iters, bo_iters, sim_iters, models, "bayesopt_1_2", true)
end

feas_model = deserialize(home_path * "models/ml_models/ga/feas_model.jls")
lam_model = deserialize(home_path * "models/ml_models/ga/lam_model.jls")
v_in_model = deserialize(home_path * "models/ml_models/ga/v_in_model.jls")
println("All models read in successfully!")

num_iters = 100
bo_iters = 1000
sim_iters = 10*60*60
# bayesopt(0.0, num_iters, bo_iters, sim_iters, "bayesopt_0_2", true)
bayesopt(0.5, num_iters, bo_iters, sim_iters, "bayesopt_5_2", true)
bayesopt(1.0, num_iters, bo_iters, sim_iters, "bayesopt_1_2", true)

###RUNS ALL EXPERIMENTS
# single_run_save()
# burden_experiments()
# bayesopt_experiments()
# medium_conditions_experiments()
println("Experiments completed!")