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
#home_path = "/home/cjmerzbacher/joint-simulator/" #for villarica runs
home_path = "C:/Users/Charlotte/OneDrive - University of Edinburgh/Documents/research/joint-simulator/"
include(home_path * "models/beta_carotene.jl")
include(home_path * "beta_carotene/bcar_sim.jl")

println("Imports completed")

#Read in saved models
feas_model = deserialize(home_path * "beta_carotene/ml_models/feas_model.jls")
lam_model = deserialize(home_path * "beta_carotene/ml_models/lam_model.jls")
v_in_model = deserialize(home_path * "beta_carotene/ml_models/v_in_model.jls")
v_fpp_model = deserialize(home_path * "beta_carotene/ml_models/v_fpp_model.jls")
v_ipp_model = deserialize(home_path * "beta_carotene/ml_models/v_ipp_model.jls")
println("All models read in successfully!")

### Run a single simulation, selecting appropriate ICs
function single_run(W, bo_iters, stable_iters, sim_iters)
    fpp_best, ipp_best, objmin, bo_data = bayesopt_ics(bo_iters, stable_iters, W)
    if objmin > 10E4
        println("No feasible ICs found for this promoter strength matrix.")
    end
    println("Initial conditions determined, starting simulation...") 
    #Run simulation with optimal ICs 
    u0 = [fpp_best, ipp_best, 0., 0., 0., 0., 0., 0., 0., 0.]
    ode_data, fba_data = fba_loop(sim_iters, W, u0, 1)
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

function lhc_w_sweep(num_iters, bo_iters, stable_iters, sim_iters, save_data=true)
    #Latin hypercube sample 1000 W values
    plan, _ = LHCoptim(1000,4,1000)
    scaled_plan = scaleLHC(plan,[(0., 0.5),(0., 0.5),(0., 0.5),(0., 0.5)])

    println("LHC Sampling complete!")

    global bo_data = DataFrame()
    global sim_fba_data = DataFrame()
    global sim_ode_data = DataFrame()
    global sum_data = DataFrame()
    
    for i in 1:num_iters
        print("Beginning iteration", i)
        W = scaled_plan[i, :]
        bo, ode, fba, sum = single_run(W, bo_iters, stable_iters, sim_iters)

        bo_data = vcat(bo_data, bo)
        sim_fba_data = vcat(sim_fba_data, fba)
        sim_ode_data = vcat(sim_ode_data, ode)
        sum_data = vcat(sum_data, sum)

        if i%10 == 0 
            if save_data
                #Save out BO data and simulation data
                CSV.write(home_path * "beta_carotene/exp_data/bo_data_1000.csv", bo_data)
                CSV.write(home_path * "beta_carotene/exp_data/sim_fba_data_1000.csv", sim_fba_data)
                CSV.write(home_path * "beta_carotene/exp_data/sim_ode_data_1000.csv", sim_ode_data)
                CSV.write(home_path * "beta_carotene/exp_data/sum_data_1000.csv", sum_data)
            end
        end
    end
    #Save out BO data and simulation data
    CSV.write(home_path * "beta_carotene/exp_data/bo_data_1000.csv", bo_data)
    CSV.write(home_path * "beta_carotene/exp_data/sim_fba_data_1000.csv", sim_fba_data)
    CSV.write(home_path * "beta_carotene/exp_data/sim_ode_data_1000.csv", sim_ode_data)
    CSV.write(home_path * "beta_carotene/exp_data/sum_data_1000.csv", sum_data)
    return bo_data, sim_fba_data, sim_ode_data, sum_data
end

num_iters = 150
bo_iters = 100
stable_iters = 500
sim_iters = 86400
bo_data, sim_ode_data, sim_fba_data, sum_data = lhc_w_sweep(num_iters, bo_iters, stable_iters, sim_iters, true)

# W = [0.2922922922922923, 0.12362362362362363, 0.05005005005005005, 0.2502502502502503]
# u0 = [0.733530, 0.014785, 0., 0., 0., 0., 0., 0., 0., 0.]
# N = 100
# ode_data, fba_data = fba_loop_noml(N, W, u0, 1)
# plot(fba_data.lam)
# plot(fba_data.v_fpp)
# plot(ode_data.v_p)