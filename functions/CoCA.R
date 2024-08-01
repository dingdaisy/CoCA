
# Computes the solution to the optimization problem in the CoCA framework:
# 
# min ||X - u v^{T}||_F^2 + rho D(v) s.t. ||u||^2 <= 1
#
# where D(v) = ||X_1 v_1 - X_2 v_2||^2 is the disagreement penalty
# between scores, using an alternating optimization algorithm.
#
# This function is part of the Cooperative Learning framework (COCA).
#
# Arguments:
# - X: A numeric matrix representing the combined data matrices, of dimension n x p.
# - i: A vector of indices partitioning variables into two views.
# - rho: A numeric value representing the weight on the agreement penalty.
# - eps: A numeric value specifying the convergence criteria (default: 1e-6).
# - maxiter: An integer specifying the maximum number of iterations (default: 100).
# - debug: A logical value indicating whether to enable debugging mode (default: FALSE).
#
# Returns:
# A list containing:
# - v: A numeric vector representing the learned coefficients of the component, of dimension p.
# - u: A numeric vector corresponding to the learned scores of the component, of dimension n.

run_coca = function(X,i,rho,eps = 1e-6,maxiter = 100,debug = F){
  
  if(debug) browser()
  
  n = nrow(X)
  p = ncol(X)
  
  # Differencing matrix
  d = rep(1,p)
  d[-i] = -1
  D = diag(d)
  
  # Initialize
  u = normalize_vector(rnorm(n))

  # Iterate until convergence
  iter = 1
  uo = rep(0,n)
  vo = v = rep(0,p)
  while( sum((uo - u)^2)/sum(u^2) > eps^2 | sum((vo - v)^2)/sum(v^2) > eps^2)
  {
    # Old to new
    uo = u; vo = v
    
    # Update v by solving a least squares problem
    Xv = rbind(diag(p), sqrt(rho)* X %*% D)
    yv = c(t(X) %*% u,rep(0,n))
    v = lm.fit(x = Xv, y = yv)$coefficients
    
    # Update u
    u = X %*% v / sqrt(sum((X %*% v)^2))
    
    iter = iter + 1
    if(iter == maxiter) break
  }
  return(list(v = v, u = u))
}

