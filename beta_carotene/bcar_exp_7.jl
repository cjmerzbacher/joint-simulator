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
home_path = "/home/cjmerzbacher/joint-simulator/" #for villarica runs
#home_path = "C:/Users/Charlotte/OneDrive - University of Edinburgh/Documents/research/joint-simulator/"
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

# #Latin hypercube sample 1000 W values
# plan, _ = LHCoptim(1000,4,1000)
# global scaled_plan = scaleLHC(plan,[(0., 0.5),(0., 0.5),(0., 0.5),(0., 0.5)])
# println("LHC Sampling complete!")


### Run a single simulation, selecting appropriate ICs
function single_run(W, bo_iters, stable_iters, sim_iters)
    fpp_best, ipp_best, objmin, bo_data = bayesopt_ics(bo_iters, stable_iters, W , 10., 10.)
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

function lhc_w_sweep(num_iters, bo_iters, stable_iters, sim_iters, save_suffix, save_data=true)
    global bo_data = DataFrame()
    global sim_fba_data = DataFrame()
    global sim_ode_data = DataFrame()
    global sum_data = DataFrame()
    
    for i in 1:num_iters
        print("Beginning iteration ", i)
        W = values(scaled_plan[i, :])[2:5]
        bo, fba, ode, sum = single_run(W, bo_iters, stable_iters, sim_iters)

        bo_data = vcat(bo_data, bo)
        sim_fba_data = vcat(sim_fba_data, fba)
        sim_ode_data = vcat(sim_ode_data, ode)
        sum_data = vcat(sum_data, sum)

        if i%5 == 0 
            if save_data
                #Save out BO data and simulation data
                CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/bo_data_"*save_suffix*".csv", bo_data)
                CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/sim_fba_data_"*save_suffix*".csv", sim_fba_data)
                CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/sim_ode_data_"*save_suffix*".csv", sim_ode_data)
                CSV.write(home_path * "beta_carotene/exp_data/sum_data_"*save_suffix*".csv", sum_data)
            end
        end
    end
    #Save out BO data and simulation data
    CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/bo_data_"*save_suffix*".csv", bo_data)
    CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/sim_fba_data_"*save_suffix*".csv", sim_fba_data)
    CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/sim_ode_data_"*save_suffix*".csv", sim_ode_data)
    CSV.write(home_path * "beta_carotene/exp_data/"*save_suffix*"/sum_data_"*save_suffix*".csv", sum_data)
    return bo_data, sim_fba_data, sim_ode_data, sum_data
end

save_suffix="long_bo_7"
num_iters = 100
bo_iters = 1000
stable_iters = 500
sim_iters = 86400
scaled_plan = CSV.read(home_path * "beta_carotene/exp_data/lhc.csv", DataFrame)
scaled_plan = scaled_plan[701:1000, :]
bo_data, sim_fba_data, sim_ode_data, sum_data = lhc_w_sweep(num_iters, bo_iters, stable_iters, sim_iters, save_suffix, true)

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