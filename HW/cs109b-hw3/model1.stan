data {
  int N; // Number of observations
  int J; // Number of districts
  int district[N];
  real urban[N];
  real living_children[N];
  real age_mean[N];
  int Y[N]; // Bernoulli Response: {0, 1} array
}

parameters {
  real mu_0;
  real<lower=0> sigma_0;
  real beta_0[J];
  real beta_1; 
  real beta_2;
  real beta_3;
}

model {
  // Prior
  mu_0 ~ normal(0, 100);
  sigma_0 ~ exponential(0.1);
  for (j in 1:J) {
    beta_0[j] ~ normal(mu_0, sigma_0);
  }
  beta_1 ~ normal(0, 100);
  beta_2 ~ normal(0, 100);
  beta_3 ~ normal(0, 100);
  
  // Likelihood
  for (i in 1:N) {
    int d = district[i];
    real p = beta_0[d] + beta_1*urban[i] + beta_2*living_children[i] + beta_3*age_mean[i];
    Y[i] ~ bernoulli_logit(p);
  }
}

generated quantities {
  int y_rep[N];         // Draws from posterior predictive dist
  for (i in 1:N) {
    int d = district[i];
    real p = beta_0[d] + beta_1*urban[i] + beta_2*living_children[i] + beta_3*age_mean[i];
    y_rep[i] = bernoulli_logit_rng(p);
  }
}

