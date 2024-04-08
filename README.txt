# Modelling dynamic host-pathway interactions at the genome scale

GitHub repository for the joint FBA-ODE simulator method described in the recent paper. Data files are too large for GitHub and available on request. To cite this work, please reference:

*Modelling dynamic host-pathway interactions at the genome scale.* by Charlotte Merzbacher, Oisin Mac Aodha, and Diego Oyarzún (2024). 

## Requirements

This code is written primarily in Julia 1.8.x and uses the following packages:
- DifferentialEquations
- COBREXA
- DataFrames
- Tulip
- Plots
- Colors
- ModelingToolkit
- Statistics
- GLM
- Random
- Flux
- ProgressMeter
- MLBase #Confusion matrix function
- Serialization
- TreeParzen
- CSV
- LatinHypercubeSampling

Visualization code is written in Python 3.x and uses the following packages: 
- pandas
- matplotlib
- seaborn
- numpy 
- colormaps 
- os
- statistics 

## Model files
1. models/beta_carotene.jl Julia implementation of beta-carotene ODE model.
2. models/glucaric_acid.jl Julia implementation of glucaric acid ODE model.
3. models/iML1515.xml SBML model of iML1515 GSM, downloaded from BIGG models database.
4. models/ml_models/ga/
	a. feas_model.jls Logistic regression model for prediction of FBA feasibility for glucaric acid
	b. lam_model.jls Linear regression model for prediction of growth rate for glucaric acid
	c. v_in_model.jls Neural network model for prediction of boundary flux for glucaric acid
5. models/ml_models/bcar/
	a. feas_model.jls Logistic regression model for prediction of FBA feasibility for beta-carotene
	b. lam_model.jls Linear regression model for prediction of growth rate for beta-carotene
	c. v_in_model.jls Linear regressionmodel for prediction of boundary flux component 1 (influx to IPP) for beta-carotene
	c. v_ipp_model.jls Linear regression model for prediction of boundary flux component 2 (efflux from IPP) for beta-carotene
	c. v_fpp_model.jls Linear regression model for prediction of boundary flux component 3 (efflux from FPP) for beta-carotene

## Experiment code  
1. experiments/bcar_experiments.jl Julia code to run all experiments with beta-carotene model.
1. experiments/ga_experiments.jl Julia code to run all experiments with glucaric model. 

Note that filepaths are at the top of the files and must be changed when repo is cloned to allow code to find necessary data files. Functions expect appropriate folders have already been created with correct names. All functions have docstrings which give information about what they do and the necessary inputs (if any).

## Visualization notebooks 
1. timing study/timing_study.ipynb Simulator runtime experiment (in Julia) and visualization (in Julia) of results for supplementary figures 1 and 2
2. visualization/bcar_visualization.ipynb Visualization code (in Python) of all results from beta-carotene case study. Includes code to generate figures 2d, 3c, d, e, f, and 4b, c.
3. visualization/ga_visualization.ipynb Visualization code (in Python) of all results from glucaric acid case study. Includes code to generate figures 2b, 3b, and 4d