# CoCA: Cooperative Component Analysis for Unsupervised Learning with Multi-view Data

Cooperative component analysis (CoCA) is a new statistical framework for unsupervised learning with multiple data views. Our framework provides a continuum of models that encompass Principal Component Analysis (PCA) and Canonical Correlation Analysis (CCA) at the two ends of the solution path. From this framework, users can select the best model that **captures significant within-view variance and strong cross-view correlation** in a data-adaptive manner. Additionally, we introduce a sparse variant of the method (sparse CoCA), which incorporates the Lasso penalty to identify key features driving observed patterns.

# Learn more about our work:
To learn more about the method, please refer to our manuscript:

Daisy Yi Ding*, Alden Green*, Min Woo Sun, and Robert Tibshirani. "CoCA: Cooperative Component Analysis." arXiv preprint arXiv:2407.16870 (2024). [Link.](https://arxiv.org/abs/2407.16870)

# Usage:
To get started with the COCA framework, follow these steps:

1. **Installation**: Ensure you have the necessary R packages (i.e. glmnet, Matrix, etc). You can install them using:
```
install.packages(c("glmnet", "Matrix"))
```
2. **Loading Functions**: Download the repository. Source the function scripts from the functions/ directory to make them available in your R environment.
```
source("functions/CoCA.R")
source("functions/sparse_CoCA.R")
source("functions/utils.R")
```
3. **Running an Example**: Here's a basic example of how to apply the sparse CoCA method.
```
# Example data matrix X with multiple views, generated from a latent factor model.

load("data/simData1.RData")
X <- simData1$X # The combined data matrix
indices <- simData1$i # Indices partitioning the variables into two views

# Penalty parameters
rho <- 0.01
lambda <- 0.01

# Apply the sparse CoCA method
fit_coca <- sparse_coca(X, indices, rho, lambda)

# Access the results
coca_coef <- result$v  # Coefficients of the identified component
```

# Directory structure:
The repository is structured as follows:
- **functions/:** Contains primary function for running CoCA and sparse CoCA.
- **examples/:** Contains example scripts to demonstrate how to use the functions.
- **simulations/**: Contains scripts for simulation studies.
- **data/**: Contains datasets used for demonstration.





 
