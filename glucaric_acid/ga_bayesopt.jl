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
home_path = "/home/cjmerzbacher/joint-simulator/" #for villarica runs
home_path = "C:/Users/Charlotte/OneDrive - University of Edinburgh/Documents/research/joint-simulator/"
include(home_path * "models/glucaric_acid.jl")
include(home_path * "glucaric_acid/ga_sim.jl")

println("Imports completed")

#Read in saved models
feas_model = deserialize(home_path * "glucaric_acid/ml_models/feas_model.jls")
lam_model = deserialize(home_path * "glucaric_acid/ml_models/lam_model.jls")
v_in_model = deserialize(home_path * "glucaric_acid/ml_models/v_in_model.jls")
println("All models read in successfully!")

### Run a single simulation, selecting appropriate ICs
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
                CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sim_fba_data_"*save_suffix*".csv", sim_fba_data)
                CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sim_ode_data_"*save_suffix*".csv", sim_ode_data)
                CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sum_data_"*save_suffix*".csv", sum_data)
                CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/bo_data_"*save_suffix*".csv", bo_data)
            end
        end

        
        #Compute objective function and save out
        return obj
    end

    #Establish search space
    space = Dict(
        :architecture => HP.Choice(:architecture, [[[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [1, 0, 0]], [[0, 1, 0], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]]]),
        :k_ino1 => HP.Uniform(:k_ino1, 10E-7, 10E-3),
        :theta_ino1 => HP.Uniform(:theta_ino1, 10E-7, 10.),
        :k_miox => HP.Uniform(:k_miox, 10E-7, 10E-3),
        :theta_miox => HP.Uniform(:theta_miox, 10E-7, 10.))
    
    #Run bayesopt
    best = fmin(objective, space, num_iters)

    if save_data
        #Save out simulation data
        CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sim_fba_data_"*save_suffix*".csv", sim_fba_data)
        CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sim_ode_data_"*save_suffix*".csv", sim_ode_data)
        CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/sum_data_"*save_suffix*".csv", sum_data)
        CSV.write(home_path * "glucaric_acid/exp_data/"*save_suffix*"/bo_data_"*save_suffix*".csv", bo_data)
    end
end

num_iters = 100
bo_iters = 1000
sim_iters = 86400
bayesopt(0.05, num_iters, bo_iters, sim_iters, "bayesopt_05", true)
bayesopt(0.1, num_iters, bo_iters, sim_iters, "bayesopt_10", true)
bayesopt(0.15, num_iters, bo_iters, sim_iters, "bayesopt_15", true)
bayesopt(0.2, num_iters, bo_iters, sim_iters, "bayesopt_20", true)