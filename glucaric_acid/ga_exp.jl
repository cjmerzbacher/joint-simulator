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
    
    return bo_data, fba_data, ode_data
end

# #Run single simulation
# A = [[0, 0, 1], [0, 0, 1]] #open loop
# W = [[2., 3.328086, 0.000041], [2., 5.070964, 0.000227]] #Optimal from MSc work - glucaric_acid_singlearch.csv
# params = [A, W]
# bo_data, fba_data, ode_data = single_run(params, 100, 86400)

# CSV.write("fba_data.csv", fba_data)
# CSV.write("ode_data.csv", ode_data)