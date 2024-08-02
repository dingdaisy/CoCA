# Example script for running sparse CoCA on a dataset generated from a latent factor model

# Source the COCA functions
source("./../functions/sparse_CoCA.R")
source("./../functions/CoCA.R")
source("./../functions/utils.R")

# Load the simulated data

############################################################################
# simData2 is a pre-generated dataset for testing the CoCA framework, 
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

load("./../data/simData2.RData")
X <- simData2$X # The combined data matrix
i <- simData2$i # Indices partitioning the variables into two views

# Fit sparse CoCA model with rho >= 0
rho <- 1e-3 # Weight on the agreement penalty
lambda <- 1e-3 # Weight on the sparsity penalty
result = sparse_coca(X, i, rho, lambda, maxiter = 100, eps = 1e-8)

# Extract the results
v = result$v  # Coefficients of the identified component

# Evaluate the estimation error
estimation_error = evaluate_mse(normalize_vector(v), normalize_vector(simData2$beta))
print(sprintf("The estimation error with rho = %.3f is %.3f", rho, estimation_error))

# Cross-validation for CoCA
nfolds = 3
nlambdas = 30 
nrhos = 30 
rhos = c(0, exp(seq(log(1e-5),log(10),length.out = nrhos)))
cv_results = sparse_coca_cv(X, rhos, nlambdas, nfolds, simData2$pu, simData2$pv)

# Plot CV curve
par(mfrow = c(1, 1))
plot_cv_coca(cv_results)

# Apply sparse CoCA with selected best parameters
result_selected = sparse_coca(X, i, cv_results$best_rho, cv_results$best_lambda)
v_selected = result_selected$v  
estimation_error_selected = evaluate_mse(normalize_vector(v_selected), normalize_vector(simData2$beta))
print(sprintf("The estimation error with rho = %.3f is %.3f", cv_results$best_rho, estimation_error_selected))



