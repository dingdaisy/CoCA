#------------------------------------------------------------------------------#
# Script to evaluate performance of CoCA without sparsity.
# The data is generated from the latent factor model:
# 
# x = (u,v) ~ (beta_u,beta_v) z  + Eta_u %*% zu  + Eta_v %*% zv  +  w, 
# 
# where
# 
# z ~ N(0,1), zu,zv ~ N(0,I_k) w ~ N(0,sigma^2 I_p), all independent.
# 
# and supp(cols(Eta_u)) \subset 1:pu, supp(cols(Eta_v)) \subset (pu + 1):p.
#------------------------------------------------------------------------------#

# Load functions
source("./../functions/coca.R")
source("sample.R")
source("./../functions/utils.R")

set.seed(9) 

# Simulation settings
niters <- 100        # Number of Monte Carlo iterations
n <- 200             # Number of observations
n_test <- 5000       # Number of test observations
pu <- 4              # Dimension of X_u (must be at least 4)
pv <- 4              # Dimension of X_v (must be at least 4)
p <- pu + pv         # Dimension of X
nrhos <- 100         # Number of rho values

# Parameters for high-variance, shared latent factor, in the given basis
beta_norm = sqrt(2)
beta = c(1,rep(0,pu - 1),1,rep(0,pv - 1))
beta = normalize_vector(beta) * beta_norm

# Parameters for high-variance, separate latent factors, in the given basis
eta_norm = beta_norm - .1 # Norm of etas
etau = c(0,1,-1,rep(0,pu  - 3),rep(0,pv))
etav = c(rep(0,pu),0,1,-1,rep(0,pv - 3))
etau = normalize_vector(etau) * eta_norm
etav = normalize_vector(etav) * eta_norm
eta = cbind(etau,etav)

# Parameters for low-variance, shared latent factors, in the given basis
u_norm = beta_norm - 1
u = c(rep(0,3),1,rep(0,pu - 4),rep(0,3),1,rep(0,pv - 4))
u = normalize_vector(u) * u_norm

# Noise covariance, in the given basis
sigma = sqrt(.09) # Standard deviation of noise in the low-noise direction
noise_cov = diag(c(rep(1,3),sigma^2,rep(1,pu - 4),rep(1,3),sigma^2,rep(1,pv - 4)))
noise_cov = noise_cov

# Choose range of disagreement penalty tuning parameters to balance
# rho D t(X) %*% X D against I.
W = cbind(beta,eta,u)
Sigma = W %*% t(W) + noise_cov      # Population covariance
d = rep(1,p); d[-(1:pu)] = -1; D = diag(d)  # Differencing matrix
e = eigen(D %*% (n * Sigma) %*% D)
rho_min = 1 / (100*max(e$values))
rho_max = 1000 / min(e$values)
rhos = exp( seq(log(rho_min),log(rho_max),length.out = nrhos) )
f = eigen(solve(Sigma) %*% D %*% Sigma %*% D)

# Initialize matrices to store results
Xpop = chol(Sigma)
loss = population_loss = pred_loss = matrix(nrow = niters, ncol = length(rhos))
recon_error = recon_error_test = matrix(nrow = niters, ncol = length(rhos))
recon_error_pop = recon_error_test_pop = matrix(nrow = niters, ncol = length(rhos))
train_loss_true_list = test_loss_true_list = c()

for(iter in 1:niters)
{
  # Sample data
  X = sample_factor_model_plus_noise(n,p,W,noise_cov)
  X_test = sample_factor_model_plus_noise(n_test,p,W,noise_cov)
  
  # Fit CoCA for different values of rho
  fits = population_fits = list()
  for(ii in 1:length(rhos))
  {
    fits[[ii]] = fit_coca(X,i = 1:pu,rho = rhos[ii],debug = F)
    population_fits[[ii]] = fit_coca(Xpop,i = 1:pu,rho = rhos[ii],debug = F)
  }
  
  for(ii in 1:length(rhos)){
    
    # Evaluate estimation error
    # Sample fits
    bh = fits[[ii]]$v
    loss[iter,ii] = evaluate_mse(normalize_vector(bh),normalize_vector(beta))
    
    # Population fits
    b = population_fits[[ii]]$v
    population_loss[iter,ii] = evaluate_mse(normalize_vector(b),normalize_vector(beta))
  }
  
  # Evaluate reconstruction error
  for(ii in 1:length(rhos)){
    bh = normalize_vector(c(fits[[ii]]$v))
    
    pred_loss[iter,ii] = sum((Sigma - bh %*% t(bh))^2)
    
    recon_error[iter,ii] = mean((proj_vector(X,bh) - X)^2)
    recon_error_test[iter,ii] = mean((proj_vector(X_test,bh) - X_test)^2)
    
    bh_pop = normalize_vector(population_fits[[ii]]$v)
    recon_error_pop[iter,ii] = mean((proj_vector(X,bh_pop) - X)^2)
    recon_error_test_pop[iter,ii] = mean((proj_vector(X_test,bh_pop) - X_test)^2)
  }
  
  # True beta
  bbeta = normalize_vector(beta)
  train_loss_true = mean((proj_vector(X, bbeta) - X)^2)
  test_loss_true = mean((proj_vector(X_test, bbeta) - X_test)^2)
  train_loss_true_list = c(train_loss_true_list, train_loss_true)
  test_loss_true_list = c(test_loss_true_list, test_loss_true)
  
  logger::log_info("Iteration ",iter," out of ",niters," complete.")
}

# Calculate average loss
ave_loss = colMeans(loss)
ave_population_loss = colMeans(population_loss)
ave_loss_recon = colMeans(recon_error_test - test_loss_true_list)
ave_loss_recon_pop = colMeans(recon_error_test_pop - test_loss_true_list)

# Plot estimation error
par(mfrow = c(1, 1))
plot(log(rhos),
     log(ave_loss),
     main = "Estimation error",
     cex.lab = 1.5,
     cex.main = 1.5,
     xlab = expression(log(rho)),
     ylab = "log(estimation error)",
     ylim = c(min(log(ave_loss))-0.1, max(log(ave_loss))))
abline(v = log(rhos)[which.min(ave_loss)],lty = 2)


# Save simulation data
simData1 <- list(
  X = X,
  i = 1:pu,
  W = W,
  beta = beta,
  noise_cov = noise_cov,
  n = n,
  pu = pu,
  pv = pv,
  rhos = rhos
)
save(simData1, file = "./../data/simData1.RData")



