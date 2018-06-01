data {
  int n_team; // Number of teams
  int N[n_team, n_team];
  int V[n_team, n_team]; // Bernoulli Response: {0, 1} array
}

parameters {
  real<lower=0> sigma;
  real lambda[n_team];
}

model {
  // Prior
  sigma ~ uniform(0, 50);
  for (t in 1:n_team) {
    lambda[t] ~ normal(0, sigma);
  }
  
  // Likelihood
  for (i in 1:n_team) {
    for (j in 1:n_team) {
      V[i, j] ~ binomial(N[i, j], exp(lambda[i])/(exp(lambda[i]) + exp(lambda[j])));
    }
  }
}

generated quantities {
  int V_rep[n_team, n_team];         // Draws from posterior predictive dist
  for (i in 1:n_team) {
    for (j in 1:n_team) {
      V_rep[i, j] = binomial_rng(N[i, j], exp(lambda[i])/(exp(lambda[i]) + exp(lambda[j])));
    }
  }
}

