
# PageRank method as described in:
# A. N. Langville and C. D. Meyer. Google's PageRank and Beyond: The Science of Search Engine 
# Rankings. Princeton University Press, 2006.
#
# PARAMETERS
# H = adjacency matrix
# v = personalization vector
# alpha = damping factor
# t = number of digits of precision
# OUTPUT
# A list with:
# pi = PageRank vector
# iter = number of iterations

pagerank = function(H, v, alpha = 0.85, t = 3) {

  n <- dim(H)[1]
  # normalize adjacency matrix by row sums and compute dangling node vector
  a <- rep(0, n)
  print(n)
  print(a)
  # row sums
  print(H)
  rs = H %*% rep(1,n)
  print(rs)
  for (i in 1:n) {
    if (rs[i] == 0) {
      a[i] = 1
    } else {
      H[i,] = H[i,] / rs[i]   
    }  
  }
  print(H)
  e = rep(1, n)
  v = rep(1.0/n, n)
  pi0 = rep(0, n)
  pi1 = rep(1/n, n)
  eps = 1/10^t
  iter = 0
  while (sum(abs(pi0 - pi1)) > eps) {
    pi0 <- pi1
    pi1 <- c(alpha) * pi1 %*% H + (c(alpha) * pi1 %*% a + (c(1) - c(alpha)) * pi1 %*% e) * v
    iter <- iter + 1
    print(pi0)
    print(pi1)
    print(eps)
    print(sum(abs(pi0 - pi1)))
  } 
  return(list(pi = as.vector(pi1), iter = iter))
}


# HITS method as described in:
# J. M. Kleinberg. Authoritative sources in a hyperlinked environment. Journal of the ACM, 
# 46(5):604-632, 1999.
#
# PARAMETERS
# L = adjacency matrix
# t = number of digits of precision
# OUTPUT
# A list with:
# a = authority vector
# h = hub vector
# val = dominant eigenvalue of authority (hub) matrix
# iter = number of iterations

hits = function(L, t = 3) {

  n = dim(L)[1]

  # compute authority matrix
  A = t(L) %*% L;

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

# Pinski & Narin method as described in:
# G. Pinski and F. Narin. Citation influence for journal aggregates of scientific publications:
# Theory, with application to the literature of physics. Information Processing & Management, 
# 12(5):297-312, 1976.
# Also Leontief closed method as described in: 
# W. W. Leontief. The Structure of American Economy, 1919-1929. Harvard University Press, 1941.
#
# PARAMETERS
# C = journal citation matrix (or Leontief coefficient matrix)
# t = number of digits of precision
# OUTPUT
# A list with:
# pi = journal influence vector (or product price vector)
# iter = number of iterations

pinski.narin = function(C, t = 3) {

  n = dim(C)[1]

  # compute matrix H
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

# Katz method as described in:
# L. Katz. A new status index derived from sociometric analysis. Psychometrika, 18:39-43, 1953.
#
# PARAMETERS
# L = adjacency matrix
# a = attenuation factor (must be lower than 1 / rho(L) for convergence)
# t = number of digits of precision
# OUTPUT
# A list with:
# pi = status vector
# iter = number of iterations

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


# Hubbell method as described in:
# C. H. Hubbell. An input-output approach to clique identification. Sociometry, 28:377-399, 1965.
#
# PARAMETERS
# W = strength matrix
# v = exogenous vector
# t = number of digits of precision
# OUTPUT
# A list with:
# pi = status vector
# iter = number of iterations

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
