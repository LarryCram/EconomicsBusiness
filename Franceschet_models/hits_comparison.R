# HITS Algorithm Implementation and Testing
#
# HITS method as described in:
# J. M. Kleinberg. Authoritative sources in a hyperlinked environment. 
# Journal of the ACM, 46(5):604-632, 1999.

# HITS function
hits = function(L, t = 3) {
  n = dim(L)[1]
  A = t(L) %*% L

  x0 = rep(0, n)
  x1 = rep(1/n, n)
  eps = 1/10^t
  iter = 0
  
  while (sum(abs(x0 - x1)) > eps) {
    x0 = x1
    x1 = A %*% x1
    m = x1[which.max(abs(x1))]
    x1 = x1 / m
    iter = iter + 1
  } 
  y = L %*% x1
  return(list(a = as.vector(x1), h = as.vector(y), val = m, iter = iter))
}

# Load data and run comparison
pagerank_data <- read.csv('data/pagerank.csv', row.names = 1)
L <- as.matrix(pagerank_data)

expected_authority <- c(
  A = 4.7, B = 45.9, C = 0.0, D = 5.3, E = 38.9, F = 5.3,
  G = 0.0, H = 0.0, I = 0.0, L = 0.0, M = 0.0
)

expected_hub <- c(
  A = 0.0, B = 0.0, C = 8.1, D = 8.9, E = 9.9, F = 14.9,
  G = 14.9, H = 14.9, I = 14.9, L = 6.8, M = 6.8
)

result <- hits(L, t = 3)

# Scale results
auth_scale <- expected_authority["B"] / result$a[2]
hub_scale <- expected_hub["G"] / result$h[7]
our_authority_scaled <- result$a * auth_scale
our_hub_scaled <- result$h * hub_scale
names(our_authority_scaled) <- rownames(L)
names(our_hub_scaled) <- rownames(L)

# Comparison and validation
auth_differences <- abs(our_authority_scaled[names(expected_authority)] - expected_authority)
hub_differences <- abs(our_hub_scaled[names(expected_hub)] - expected_hub)
auth_mae <- mean(auth_differences)
hub_mae <- mean(hub_differences)
tolerance <- 0.1
auth_close <- auth_differences <= tolerance
hub_close <- hub_differences <= tolerance

cat("Mean Absolute Error (Authority):", round(auth_mae, 3), "\n")
cat("Mean Absolute Error (Hub):", round(hub_mae, 3), "\n")
cat("Authority nodes within tolerance (±", tolerance, "):", sum(auth_close), "out of", length(auth_close), "\n")
cat("Hub nodes within tolerance (±", tolerance, "):", sum(hub_close), "out of", length(hub_close), "\n")
cat(if (all(auth_close) && all(hub_close)) "✓ Perfect match!\n" else "✗ Some discrepancies found.\n")
