#------------------------------------------------------------------------------#
# Sample from the latent factor model
# 
# x = (u,v) ~ (beta_u,beta_v) z  + Eta_u %*% zu  + Eta_v %*% zv  +  w, 
# 
# where
# 
# z ~ N(0,1), zu,zv ~ N(0,I_k) w ~ N(0,sigma^2 I_p), all independent.
#------------------------------------------------------------------------------#

sample_factor_model_plus_noise = function(n,p,W,noise_cov){

  # Sample random vectors from a latent factor model with additive noise
  #
  # Args:
  #   n: An integer representing the number of samples.
  #   p: An integer representing the number of variables.
  #   W: A matrix representing the factor loadings.
  #   noise_cov: A covariance matrix for the noise.
  #
  # Returns:
  #   A matrix where each row is a sample from the latent factor model with noise.
  
  
  # Factors
  k = ncol(W)
  Z = matrix(rnorm(n*k),ncol = k,nrow = n)
  Xfactor = Z %*% t(W)
  
  # Noise
  rtnoisecov = chol(noise_cov)
  Xnoise = matrix(rnorm(n*p),nrow = n,ncol = p) %*% t(rtnoisecov)
  
  # X = factors + noise
  X = Xfactor + Xnoise
  X = as.matrix(X)
  return(X)
}
