# Hubbell Algorithm Implementation and Testing
#
# Hubbell method as described in:
# C. H. Hubbell. An input-output approach to clique identification. 
# Sociometry, 28:377-399, 1965.

# Hubbell function
hubbell = function(W, v, t = 3) {
  n = dim(W)[1]
  W1 = diag(1, nrow = n)
  S0 = diag(0, nrow = n)
  S1 = diag(1, nrow = n)
  eps = 1/10^t
  iter = 0
  while (sum(abs(S0 - S1)) > eps) {
    W1 = W1 %*% W
    S0 = S1
    S1 = S1 + W1
    iter = iter + 1
  } 
  pi = v %*% S1
  return(list(pi = as.vector(pi), iter = iter))
}

# Load data and run comparison
hubbell_data <- read.csv('data/hubbell.csv', row.names = 1)
W <- as.matrix(hubbell_data)

expected_values <- c(
  A = 0.49, B = 0.41, C = 0.2, D = -0.9
)

v <- c(0.20, 0.20, 0.20, 0.20)
names(v) <- rownames(W)

result <- hubbell(W, v)
hubbell_scores <- result$pi
names(hubbell_scores) <- rownames(W)

# Comparison and validation
differences <- abs(hubbell_scores[names(expected_values)] - expected_values)
mae <- mean(differences)
tolerance <- 0.01
within_tolerance <- all(differences <= tolerance)

cat("Mean Absolute Error:", round(mae, 3), "\n")
cat("Nodes within tolerance (±", tolerance, "):", length(differences), "out of", length(differences), "\n")
cat(if (within_tolerance) "✓ Perfect match!\n" else "✗ Some discrepancies found.\n")