
# Helper functions for CoCA

# Normalizes a vector to have a unit norm.
normalize_vector = function(v) v/sqrt(sum(v^2))

# Projects a matrix onto the subspace defined by a vector.
proj_vector = function(X,v){
  
  ##############################################################
  # Args:
  #   X: A numeric matrix where each row is an observation.
  #   v: A numeric vector onto which the projection is performed.
  # Returns:
  #   A numeric matrix representing the projection of X onto the subspace spanned by v.
  ##############################################################
  
  v = normalize_vector(v)
  X %*% v %*% t(v)
} 

# Computes the Euclidean (L2) norm of a vector.
twoNorm <- function(x){
  sqrt(sum(x^2))
}

# Computes the Mean Squared Error (MSE) between the estimated factor and the true factor.
evaluate_mse <- function(bh, beta) {
  
  ##############################################################
  # Args:
  #   bh: A numeric vector representing the estimated factor.
  #   beta: A numeric vector representing the true factor.
  #
  # Returns:
  #   A numeric value representing the MSE between the estimated factor (bh) and the true factor (beta).
  #   The function ensures that the comparison accounts for the sign ambiguity by taking the minimum
  #   of the squared differences between bh and beta, and bh and -beta.
  ##############################################################
  
  min(sum((bh - beta)^2), sum((-bh - beta)^2))
}

