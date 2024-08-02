
source("sample.R")

# Simulation suite
seed_each <- 5
set.seed(seed_each)
n <- 200      # Number of observations
pu <- 30      # Dimension of X_u
pv <- 30      # Dimension of X_v
p <- pu + pv  # Dimension of X
k <- 1        # Number of separate latent factors (must be at most min(pu, pv) - 1)
s <- 6        # Sparsity of the leading eigenvector (must be an even number, for symmetry across views)
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

X = sample_factor_model_plus_noise(n,p,W,noise_cov)

# Save simulation data
simData2 <- list(
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
save(simData2, file = "./../data/simData2.RData")




