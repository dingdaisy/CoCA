# Example script for running CoCA on a dataset generated from a latent factor model

# Source the COCA functions
source("./../functions/CoCA.R")
source("./../functions/utils.R")

# Load the simulated data

############################################################################
# simData1 is a pre-generated dataset for testing the CoCA framework, 
# generated from the latent factor model:
#
# x = (u,v) ~ (beta_u,beta_v) z  + Eta_u %*% zu  + Eta_v %*% zv  +  w, 
# 
# where
# 
# z ~ N(0,1), zu,zv ~ N(0,I_k) w ~ N(0,sigma^2 I_p), all independent.
# 
# and supp(cols(Eta_u)) \subset 1:pu, supp(cols(Eta_v)) \subset (pu + 1):p.
############################################################################

load("./../data/simData1.RData")
X <- simData1$X # The combined data matrix
i <- simData1$i # Indices partitioning the variables into two views

# Fit CoCA model with rho >= 0
rho <- 0.1 # Weight on the agreement penalty
result = fit_coca(X, i, rho, maxiter = 100, eps = 1e-6)

# Extract the results
v = result$v  # Coefficients of the identified component

# Evaluate the estimation error
estimation_error = evaluate_mse(normalize_vector(v), normalize_vector(simData1$beta))
print(sprintf("The estimation error with rho = %.2f is %.3f", rho, estimation_error))






