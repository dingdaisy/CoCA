
library(Matrix)
library(glmnet)

# Computes the solution to the optimization problem in the SPARSE CoCA framework:
# 
# min ||X - u v^{T}||_F^2 + rho D(v) + lambda |v|_1 s.t. ||u||^2 <= 1
#
# where D(v) = ||X_1 v_1 - X_2 v_2||^2 is the disagreement penalty
# between scores, using an alternating optimization algorithm with lasso regularization.
#
# This function is part of the SPARSE Cooperative Learning framework (COCA) framework.
#
# Arguments:
#   X: A numeric matrix representing the combined data matrices, of dimension n x p.
#   i: A vector of indices partitioning variables into two views.
#   rho: A numeric value representing the weight on the agreement penalty.
#   lambda: A numeric value representing the lasso penalty parameter on the L1 norm of v.
#   start: An optional numeric vector for the initial value of u (default: NULL).
#   eps: A numeric value specifying the convergence criteria (default: 1e-8).
#   maxiter: An integer specifying the maximum number of iterations (default: 100).
#   debug: A logical value indicating whether to enable debugging mode (default: FALSE).
#
# Returns:
#   A list containing:
# - v: A numeric vector representing the learned coefficients of the component, of dimension p.
# - u: A numeric vector corresponding to the learned scores of the component, of dimension n.

sparse_coca = function(X,i,rho,lambda = 0,start = NULL,eps = 1e-8,maxiter = 100,debug = F){
  
  if(debug) browser()
  
  n = nrow(X)
  p = ncol(X)
  
  # Differencing matrix
  d = rep(1,p)
  d[-i] = -1
  D = Diagonal(x = d)
  
  # Initialize
  if(!is.null(start)){
    u = start
  } else {
    u = normalize_vector(rnorm(n))
  }
  
  # Iterate until convergence
  iter = 1
  uo = rep(0,n); vo = v = rep(0,p)
  while( sum((uo - u)^2)/sum(u^2) > eps^2 | sum((vo - v)^2)/sum(v^2) > eps^2)
  {
    # Old to new
    uo = u; vo = v
    
    # Update v by solving a lasso problem
    Xv = rbind(Diagonal(p), sqrt(rho)* X %*% D)
    yv = c(t(X) %*% u,rep(0,n))
    v = glmnet::glmnet(x = Xv, y = yv, lambda = lambda,intercept = F,standardize = F)$beta[,1]
    
    # Update u
    if(all(v == 0)){u = NULL; break}
    u = X %*% v / sqrt(sum((X %*% v)^2))
    
    iter = iter + 1
    if(iter == maxiter) break
  }

  return(list(v = v, u = u))
}

# Perform cross-validation for CoCA to select the best parameters.
# The data matrix X is split into nfolds, and for each fold, the model 
# is trained on the training set and evaluated on the validation set for reconstruction error.. 
# The mean squared error is computed for each combination 
# of rho and lambda, and the results are averaged across folds.
#
# Note: There are different options for CV procedures to determine the optimal values 
# of hyperparameters rho and lambda in CoCA, including K-fold CV and "speckled CV" for 
# unsupervised settings, and K-fold CV for supervised settings. 
# We implement here the K-fold CV.
#
# Args:
#   X: A combined data matrix.
#   rhos: A numeric vector of rho values (weights on the agreement penalty).
#   n_lambdas: An integer representing the number of lambda values to select from.
#   nfolds: An integer representing the number of folds for cross-validation.
#   pu: An integer representing the dimension of the first data view.
#   pv: An integer representing the dimension of the second data view.
#   eps: A numeric value representing the convergence criterion.
#
# Returns:
#   A list containing:
#     results: An array of cross-validation errors for each combination of rho and lambda across all folds.
#     mean_errors: A matrix of mean cross-validation errors for each combination of rho and lambda.
#     median_errors: A matrix of median cross-validation errors for each combination of rho and lambda.
#     lambdas: A matrix of lambda sequences for each rho value.
#     best_rho: The rho value with the minimum mean error.
#     best_lambda: The lambda value with the minimum mean error.
#     best_rho_median: The rho value with the minimum median error.
#     best_lambda_median: The lambda value with the minimum median error.

sparse_coca_cv <- function(X, rhos, n_lambdas, nfolds, pu, pv, eps = 1e-8, maxiter = 100) {
  
  # Create folds for cross-validation
  folds <- cut(seq(1, nrow(X)), breaks = nfolds, labels = FALSE)
  lambdas <- matrix(nrow = length(rhos), ncol = n_lambdas)
  results <- array(dim = c(length(rhos), n_lambdas, nfolds))
  p = pu + pv
  n = nrow(X)
  
  # Loop over each rho value
  for (ii in 1:length(rhos)) {
    fits = list()
    
    # Loop over each fold
    for (k in 1:nfolds) {
      
      # Split data into training and validation
      train_idx <- which(folds != k)
      valid_idx <- which(folds == k)
      X_train <- X[train_idx, ]
      X_valid <- X[valid_idx, ]
      
      # Choose lambda sequence during the first fold
      if (k == 1){
        # Choose lambda as would be done in first step of alternating algorithm,
        # initialized at the non-sparse solution to CoCA.
        # Here we use X for choosing lambda.
        # Only do this once for the first fold split and use the same sequence for the rest.
        nosparse = fit_coca(X,i = 1:pu,rho = rhos[ii],debug = F,eps = eps, maxiter = maxiter)
        u = nosparse$u
        d = rep(1,p); d[-(1:pu)] = -1; D = diag(d)
        Xv = rbind(Diagonal(p), sqrt(rhos[ii])* X %*% D)
        yv = c(t(X) %*% u,rep(0,n))
        
        # Fit glmnet to obtain the lambda sequence.
        lambdaseq = glmnet::glmnet(x = Xv, y = yv, nlambda = n_lambdas - 1,intercept = F,standardize = F)$lambda
        
        # Pad the lambda sequence with 0s to make it the desired length.
        if (length(lambdaseq) < n_lambdas) lambdaseq = c(lambdaseq,rep(0,n_lambdas - length(lambdaseq)))
        lambdas[ii,] = lambdaseq / 20
      }
      
      # Fit non-sparse CoCA on the training data
      nosparse_fold = fit_coca(X_train,i = 1:pu,rho = rhos[ii],debug = F,eps = eps, maxiter = maxiter)
      
      # Loop over each lambda value
      for (jj in 1:n_lambdas) {
        # Warm start the fit with the previous solution
        start <- if (jj == 1) nosparse_fold$u else fits[[jj - 1]]$u
        
        # Fit sparse CoCA on the training data
        fits[[jj]] <- sparse_coca(X_train, i = 1:pu, rho = rhos[ii], lambda = lambdas[ii, jj], start = start, eps = eps, maxiter = maxiter, debug = F)
        
        # Evaluate on validation data for reconstruction error
        fold_errors <- mean((proj_vector(X_valid, normalize_vector(fits[[jj]]$v)) - X_valid)^2) 
        results[ii, jj, k] <- fold_errors
      }
    }
  }
  
  # Calculate mean errors across all folds
  mean_errors <- apply(results, c(1, 2), mean)
  median_errors <- apply(results, c(1, 2), median)
  
  # Find the best rho and lambda based on the minimum mean/median error
  min_error_idx <- which(mean_errors == min(mean_errors, na.rm=TRUE), arr.ind = TRUE)
  best_rho_mean <- rhos[min_error_idx[1]]
  best_lambda_mean <- lambdas[min_error_idx[2]]
  
  min_error_idx_median <- which(median_errors == min(median_errors, na.rm=TRUE), arr.ind = TRUE)
  best_rho_median <- rhos[min_error_idx_median[1]]
  best_lambda_median <- lambdas[min_error_idx_median[2]]
  
  # Return the results
  list(results = results,
       mean_errors = mean_errors, 
       median_errors = median_errors,
       lambdas = lambdas,
       rhos = rhos,
       best_rho = best_rho_mean,
       best_lambda = best_lambda_mean,
       best_rho_median = best_rho_median,
       best_lambda_median = best_lambda_median)
}

# Plots the cross-validation errors as a function of rho.
#
# Args:
#   coca_results: A list containing the results of sparse_coca_cv.
#   by: A character string indicating whether to summarize by 'mean_error' or 'median_error'.
#
# Returns:
#   A plot of the cross-validation errors.

plot_cv_coca <- function(coca_results, by = "mean_error") {
  if (by == "mean_error") {
    ave_loss_by_rho = apply(coca_results$mean_errors,1,min)
    plot(log(coca_results$rhos), log(ave_loss_by_rho), 
         main = "CV Reconstruction Error\n(Summarized by Mean)", 
         xlab = expression(log(rho)), 
         ylab = "log(Reconstruction error)",
         pch = 19, # Solid circle
         cex = 0.7,  # Size of points
         col = "black")
    
  } else if (by == "median_error") {
    ave_loss_by_rho = apply(coca_results$median_errors,1,min)
    plot(log(coca_results$rhos), log(ave_loss_by_rho), 
         main = "CV Reconstruction Error\n(Summarized by Median)", 
         xlab = expression(log(rho)), 
         ylab = "log(Reconstruction error)",
         pch = 19, # Solid circle
         cex = 0.7,  # Size of points
         col = "black")
  } else {
    stop("Invalid 'by' argument. Please specify 'mean_error' or 'median_error'.")
  }
}

