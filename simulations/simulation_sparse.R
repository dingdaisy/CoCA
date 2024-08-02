#------------------------------------------------------------------------------#
# Script to evaluate performance of Sparse CoCA.
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

library(Matrix)
library(glmnet)

# Load functions
source("./../functions/sparse_CoCA.R")
source("./../functions/CoCA.R")
source("sample.R")
source("./../functions/utils.R")

# Simulation suite
seed_each <- 12
set.seed(seed_each)
niters <- 10
n <- 200      # Number of observations
pu <- 30      # Dimension of X_u
pv <- 30      # Dimension of X_v
p <- pu + pv  # Dimension of X
k <- 1        # Number of separate latent factors (must be at most min(pu, pv) - 1)
s <- 2        # Sparsity of the leading eigenvector (must be an even number, for symmetry across views)
nrhos <- 50
nlambdas <- 50
eps <- 1e-4
n_val <- 200
n_test <- 5000

# Parameters for high-variance, shared latent factor
beta_norm = sqrt(2)
beta = c(rep(1,s/2),rep(0,pu - s/2),rep(1,s/2),rep(0,pv - s/2))
beta = normalize_vector(beta) * beta_norm
beta = beta

# Parameters for high-variance, separate latent factors
eta_norm = beta_norm - 0.1 # Norm of etas
etau = c(rep(0,s/2),rep(1,s),rep(0,pu - 3*s/2),rep(0,pv))
etav = c(rep(0,pv),rep(0,s/2),rep(1,s),rep(0,pv - 3*s/2))
etau = normalize_vector(etau) * eta_norm
etav = normalize_vector(etav) * eta_norm
eta = cbind(etau,etav)
eta = eta

# Parameters for low-variance, shared latent factors
u_norm = beta_norm - 1
u = c(rep(0,3*s/2),rep(1,s/2),rep(0,pu - 2*s),rep(0,3*s/2),rep(1,s/2),rep(0,pv - 2*s))
u = normalize_vector(u) * u_norm
u = u

# Noise covariance, in the given basis
sigma = sqrt(.09) # Standard deviation of noise in the low-noise direction
noise_cov = Diagonal(x = c(rep(1,3*s/2),rep(sigma^2,s/2),rep(1,pu - 2*s),
                   rep(1,3*s/2),rep(sigma^2,s/2),rep(1,pu - 2*s)))
noise_enlarge = 1
noise_cov =  noise_cov * noise_enlarge

# Choose range of disagreement penalty tuning parameters to balance
# rho D t(X) %*% X D against I.
W = cbind(beta,eta,u)
Sigma = W %*% t(W) + noise_cov      # Population covariance
d = rep(1,p); d[-(1:pu)] = -1; D = diag(d)  # Differencing matrix
e = eigen(D %*% (n * Sigma) %*% D)
rho_min = 1 / (100*max(e$values))
rho_max = 1000 / min(e$values)
rhos = exp( seq(log(rho_min),log(rho_max),length.out = nrhos) )
Xpop = as.matrix(chol(Sigma))

# Initialize matrices to store results
loss = pred_loss = l1norm = disagreement = train_loss = val_loss = test_loss = array(dim = c(niters,nrhos,nlambdas))
train_loss_cca_list = val_loss_cca_list = test_loss_cca_list = c()
train_loss_pca_list = val_loss_pca_list = test_loss_pca_list = c()
train_loss_true_list = val_loss_true_list = test_loss_true_list = c()
population_loss = matrix(nrow = niters, ncol = nrhos)
lambdas = matrix(nrow = nrhos, ncol = nlambdas)

for(iter in 1:niters)
{
  # Sample data
  X = sample_factor_model_plus_noise(n,p,W,noise_cov)
  X_val = sample_factor_model_plus_noise(n_val,p,W,noise_cov)
  X_test = sample_factor_model_plus_noise(n_test,p,W,noise_cov)
  
  # Fit Sparse CoCA for different values of rho, lambda
  fits = population_fits = list()
  for(ii in 1:length(rhos))
  {
    nosparse = fit_coca(X,i = 1:pu,rho = rhos[ii],debug = F,eps = eps)
    
    # In the first iteration, choose lambda sequence
    if(iter == 1)
    {
      # Choose lambda as would be done in first step of alternating algorithm,
      # initialized at the non-sparse solution to CoCA.
      u = nosparse$u
      Xv = rbind(Diagonal(p), sqrt(rhos[ii])* X %*% D)
      yv = c(t(X) %*% u,rep(0,n))
      lambdaseq = glmnet::glmnet(x = Xv, y = yv, nlambda = nlambdas - 1,intercept = F,standardize = F)$lambda
      
      # Pad the lambda sequence with 0s to make it the desired length.
      if(length(lambdaseq) < nlambdas) lambdaseq = c(lambdaseq,rep(0,nlambdas - length(lambdaseq)))
      lambdas[ii,] = lambdaseq
    }
    
    fits[[ii]] = list()
    for(jj in 1:nlambdas)
    {
      # Fit for each lambda warm started at the previous fit
      if(jj == 1) start = nosparse$u else start = fits[[ii]][[jj - 1]]$u
      fits[[ii]][[jj]] = sparse_coca(X,i = 1:pu,
                                                rho = rhos[ii],lambda = lambdas[ii,jj],
                                                start = start,
                                                eps = eps, debug = F)
    }
    
    # Population fits
    population_fits[[ii]] = fit_coca(Xpop,i = 1:pu,rho = rhos[ii],debug = F)
  }
  
  # Evaluate estimation error (to true leading eigenvector), reconstruction error,
  # and l1 norm.
  for(ii in 1:length(rhos)){
    
    for(jj in 1:nlambdas)
    {
      # Eigenvector estimation error
      v = as.numeric(fits[[ii]][[jj]]$v)
      bh = normalize_vector(v)
      loss[iter,ii,jj] = evaluate_mse(bh,normalize_vector(beta))
      
      # Covariance matrix reconstruction error
      pred_loss[iter,ii,jj] = sum((Sigma - bh %*% t(bh))^2)
      
      # l1 norm
      l1norm[iter,ii,jj] = sum(abs(bh))
      
      # disagreement
      disagreement[iter,ii,jj] = mean((X[,1:pu] %*% bh[1:pu] - X[,-(1:pu)] %*% bh[-(1:pu)])^2)
      
      train_loss[iter,ii,jj] = mean((proj_vector(X,bh) - X)^2)
      val_loss[iter,ii,jj] = mean((proj_vector(X_val,bh) - X_val)^2)
      test_loss[iter,ii,jj] = mean((proj_vector(X_test,bh) - X_test)^2)
    }
    
    # Estimation error of population fits
    b = population_fits[[ii]]$v
    population_loss[iter,ii] = evaluate_mse(normalize_vector(b),normalize_vector(beta))
  }
  
  # True beta
  bbeta = normalize_vector(beta)
  train_loss_true = mean((proj_vector(X, bbeta) - X)^2)
  val_loss_true = mean((proj_vector(X_val, bbeta) - X_val)^2)
  test_loss_true = mean((proj_vector(X_test, bbeta) - X_test)^2)
  train_loss_true_list = c(train_loss_true_list, train_loss_true)
  val_loss_true_list = c(val_loss_true_list, val_loss_true)
  test_loss_true_list = c(test_loss_true_list, test_loss_true)
  
  logger::log_info("Iteration ",iter," out of ",niters," complete.")
}

# Aggregate estimation eror 
ave_loss = list()
ave_loss$mean = colMeans(loss, na.rm=TRUE)
ave_loss$median = apply(loss,c(2,3),median, na.rm=TRUE)
ave_loss_by_rho = apply(ave_loss$median,1,min)

# Plot estimation error
par(mfrow = c(1, 1))
plot(log(rhos), log(ave_loss_by_rho), 
     main = "Estimation Error\n(Summarized by Median)", 
     xlab = expression(log(rho)), 
     ylab = "log(Estimation error)",
     pch = 19, # Solid circle
     cex = 0.7,  # Size of points
     col = "black")

# Add vertical dashed line at the minimum log_ave_loss_by_rho
abline(v = log(rhos)[which.min(ave_loss_by_rho)], col = "black", lty = 2)


