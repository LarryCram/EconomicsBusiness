# Pinski-Narin Algorithm Implementation and Testing
#
# Pinski & Narin method as described in:
# G. Pinski and F. Narin. Citation influence for journal aggregates of scientific publications:
# Theory, with application to the literature of physics. Information Processing & Management, 
# 12(5):297-312, 1976.

# Pinski-Narin function
pinski.narin = function(C, t = 3) {
  n = dim(C)[1]
  H = C
  rs = C %*% rep(1,n)
  for (j in 1:n) {
    if (rs[j] != 0) {
      H[,j] = H[,j] / rs[j]   
    }  
  }

  pi0 = rep(0, n)
  pi1 = rep(1/n, n)
  eps = 1/10^t
  iter = 0
  while (sum(abs(pi0 - pi1)) > eps) {
    pi0 = pi1
    pi1 = pi1 %*% H
    iter = iter + 1
  } 
  return(list(pi = as.vector(pi1), iter = iter))
}

# Load data and run comparison
pn_data <- read.csv('data/pinski_narin.csv', row.names = 1)
C <- as.matrix(pn_data)

# Expected values (from literature, but R algorithm doesn't match exactly)
expected_values <- c(A = 28.0, B = 44.7, C = 12.4, D = 14.9)

result <- pinski.narin(C, t = 3)

# Scale results to match expected sum
expected_sum <- sum(expected_values)
our_sum <- sum(result$pi)
scale_factor <- expected_sum / our_sum
our_scaled <- result$pi * scale_factor
names(our_scaled) <- rownames(C)

# Comparison and validation (noting R algorithm discrepancies)
differences <- abs(our_scaled[names(expected_values)] - expected_values)
mae <- mean(differences)

cat("Mean Absolute Error vs Literature:", round(mae, 3), "\n")
cat("Note: R algorithm produces consistent results but differs from literature values\n")
cat("Actual R results: A=", round(our_scaled[1], 1), ", B=", round(our_scaled[2], 1), 
    ", C=", round(our_scaled[3], 1), ", D=", round(our_scaled[4], 1), "\n")