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
#home_path = "C:/Users/Charlotte/OneDrive - University of Edinburgh/Documents/research/joint-simulator/"
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

    #Create summary data frame with delta lam, final lam, overall promoter strength, four W values, total beta-carotene production
    summary = DataFrame("k_ino1" => [W[1][3]], "theta_ino1" => [W[1][2]], "k_miox" => [W[2][3]], "theta_miox" => [W[2][2]], "ga_tot" => [sum(ode_data.mi)], "objmin" => [objmin])
    
    return bo_data, fba_data, ode_data, summary
end

function new_name(A, Ws, num_iters, bo_iters, sim_iters, save_suffix, save_data=true)
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

# A = [[0, 0, 1], [0, 0, 1]] #open loop
# W = [[2., 3.328086, 0.000041], [2., 5.070964, 0.000227]] #Optimal from MSc work - glucaric_acid_singlearch.csv
# W = [[2.0, 7.070183513988823, 0.09],
# [2.0, 4.397481247882533, 0.8887804288102178]]
# params = [A, W]
# bo_data, fba_data, ode_data, sum_data = single_run(params, 1000, 100)

# println("OPEN LOOP CONTROL")
# A = [[0, 0, 1], [0, 0, 1]] #open loop
# arch = "nc"
# save_suffix=arch
# num_iters = 400
# bo_iters = 1000
# sim_iters = 86400

# scaled_plan = CSV.read(home_path * "glucaric_acid/exp_data/"*arch*"_lhc.csv", DataFrame)
# scaled_plan = scaled_plan[101:500, :]
# bo_data, sim_fba_data, sim_ode_data, sum_data = new_name(A, scaled_plan, num_iters, bo_iters, sim_iters, save_suffix, true)

# println("DUAL CONTROL")
# A = [[0, 0, 1], [0, 0, 1]] #open loop
# arch = "dc"
# save_suffix=arch
# num_iters = 400
# bo_iters = 1000
# sim_iters = 86400

# scaled_plan = CSV.read(home_path * "glucaric_acid/exp_data/"*arch*"_lhc.csv", DataFrame)
# scaled_plan = scaled_plan[101:500, :]
# bo_data, sim_fba_data, sim_ode_data, sum_data = new_name(A, scaled_plan, num_iters, bo_iters, sim_iters, save_suffix, true)

# println("UPSTREAM REPRESSION")
# A = [[0, 1, 0], [0, 0, 1]] 
# arch = "ur"
# save_suffix=arch
# num_iters = 500
# bo_iters = 1000
# sim_iters = 86400

# scaled_plan = CSV.read(home_path * "glucaric_acid/exp_data/"*arch*"_lhc.csv", DataFrame)
# bo_data, sim_fba_data, sim_ode_data, sum_data = new_name(A, scaled_plan, num_iters, bo_iters, sim_iters, save_suffix, true)

println("DOWNSTREAM ACTIVATION")
A = [[0, 0, 1], [1, 0, 0]] 
arch = "da"
save_suffix=arch
num_iters = 500-55
bo_iters = 1000
sim_iters = 86400

scaled_plan = CSV.read(home_path * "glucaric_acid/exp_data/"*arch*"_lhc.csv", DataFrame)
scaled_plan = scaled_plan[56:500, :]
bo_data, sim_fba_data, sim_ode_data, sum_data = new_name(A, scaled_plan, num_iters, bo_iters, sim_iters, save_suffix, true)
