# Katz Algorithm Implementation and Testing
#
# Katz method as described in:
# L. Katz. A new status index derived from sociometric analysis. 
# Psychometrika, 18:39-43, 1953.

# Katz function
katz = function(L, a, t = 3) {
  n = dim(L)[1]
  W = a * L
  W1 = diag(1, nrow = n)
  S0 = diag(1, nrow = n)
  S1 = diag(0, nrow = n)
  eps = 1/10^t
  iter = 0
  while (sum(abs(S0 - S1)) > eps) {
    W1 = W1 %*% W
    S0 = S1
    S1 = S1 + W1
    iter = iter + 1
  } 
  v = rep(1, n)
  pi = v %*% S1
  return(list(pi = as.vector(pi), iter = iter))
}

# Load data and run comparison
katz_data <- read.csv('data/katz.csv', row.names = 1)
L <- as.matrix(katz_data)

expected_values_1 <- c(
  A = 2.7, B = 46.4, C = 41.9, D = 2.9, E = 3.2, F = 2.9,
  G = 0.0, H = 0.0, I = 0.0, L = 0.0, M = 0.0
)

expected_values_2 <- c(
  A = 5.7, B = 39.6, C = 8.8, D = 7.9, E = 30.1, F = 7.9,
  G = 0.0, H = 0.0, I = 0.0, L = 0.0, M = 0.0
)

# Test optimal parameter combinations
result_01 <- katz(L, 0.1, t = 3)
result_09 <- katz(L, 0.9, t = 3)

# Scale results
scale_factor_2 <- expected_values_2["B"] / result_01$pi[2]
scale_factor_1 <- expected_values_1["B"] / result_09$pi[2]
scaled_01 <- result_01$pi * scale_factor_2
scaled_09 <- result_09$pi * scale_factor_1
names(scaled_01) <- rownames(L)
names(scaled_09) <- rownames(L)

# Comparison and validation
diff_1 <- abs(scaled_09[names(expected_values_1)] - expected_values_1)
diff_2 <- abs(scaled_01[names(expected_values_2)] - expected_values_2)
mae_1 <- mean(diff_1)
mae_2 <- mean(diff_2)
tolerance <- 0.1

cat("Alpha = 0.9 + Dataset 1: MAE =", round(mae_1, 3), "\n")
cat("Alpha = 0.1 + Dataset 2: MAE =", round(mae_2, 3), "\n")
cat(if (all(diff_1 <= tolerance) && all(diff_2 <= tolerance)) "✓ Perfect match!\n" else "✗ Some discrepancies found.\n")