# PageRank Algorithm Implementation and Testing
#
# PageRank method as described in:
# A. N. Langville and C. D. Meyer. Google's PageRank and Beyond: The Science of Search Engine 
# Rankings. Princeton University Press, 2006.

# PageRank function
pagerank = function(H, v = NULL, alpha = 0.85, t = 3) {
  n <- dim(H)[1]
  a <- rep(0, n)
  rs = H %*% rep(1,n)
  
  for (i in 1:n) {
    if (rs[i] == 0) {
      a[i] = 1
    } else {
      H[i,] = H[i,] / rs[i]   
    }  
  }
  
  e = rep(1, n)
  if (is.null(v)) {
    v = rep(1.0/n, n)
  }
  pi0 = rep(0, n)
  pi1 = rep(1/n, n)
  eps = 1/10^t
  iter = 0
  
  while (sum(abs(pi0 - pi1)) > eps) {
    pi0 <- pi1
    pi1 <- c(alpha) * pi1 %*% H + (c(alpha) * pi1 %*% a + (c(1) - c(alpha)) * pi1 %*% e) * v
    iter <- iter + 1
  } 
  return(list(pi = as.vector(pi1), iter = iter))
}

# Load data and run comparison
pagerank_data <- read.csv('data/pagerank.csv', row.names = 1)
H <- as.matrix(pagerank_data)

expected_values <- c(
  A = 3.3, B = 38.4, C = 34.3, D = 3.9, E = 8.1, F = 3.9,
  G = 1.6, H = 1.6, I = 1.6, L = 1.6, M = 1.6
)

result <- pagerank(H, alpha = 0.85, t = 3)
our_results_x100 <- result$pi * 100
names(our_results_x100) <- rownames(H)

# Comparison and validation
comparison_df <- data.frame(
  Node = names(expected_values),
  Expected = expected_values,
  Our_Result = round(our_results_x100[names(expected_values)], 1),
  Difference = round(our_results_x100[names(expected_values)] - expected_values, 1),
  Percent_Error = round(abs(our_results_x100[names(expected_values)] - expected_values) / expected_values * 100, 2)
)

mae <- mean(abs(comparison_df$Difference))
tolerance <- 0.1
close_matches <- abs(comparison_df$Difference) <= tolerance

cat("Mean Absolute Error:", round(mae, 3), "\n")
cat("Nodes within tolerance (±", tolerance, "):", sum(close_matches), "out of", length(close_matches), "\n")
cat(if (all(close_matches)) "✓ Perfect match!\n" else "✗ Some discrepancies found.\n")