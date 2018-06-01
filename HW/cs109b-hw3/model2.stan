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
  real<lower=0> sigma_1;
  real<lower=0> sigma_2;
  real<lower=0> sigma_3;
  real beta_0[J];
  real beta_1[J]; 
  real beta_2[J];
  real beta_3[J];
}

model {
  // Prior
  mu_0 ~ normal(0, 100);
  sigma_0 ~ exponential(0.1);
  sigma_1 ~ exponential(0.1);
  sigma_2 ~ exponential(0.1);
  sigma_3 ~ exponential(0.1);
  for (j in 1:J) {
    beta_0[j] ~ normal(mu_0, sigma_0);
    beta_1[j] ~ normal(0, sigma_1);
    beta_2[j] ~ normal(0, sigma_2);
    beta_3[j] ~ normal(0, sigma_3);
  }
  
  // Likelihood
  for (i in 1:N) {
    int d = district[i];
    real p = beta_0[d] + beta_1[d]*urban[i] + beta_2[d]*living_children[i] + beta_3[d]*age_mean[i];
    Y[i] ~ bernoulli_logit(p);
  }
}

generated quantities {
  int y_rep[N];         // Draws from posterior predictive dist
  for (i in 1:N) {
    int d = district[i];
    real p = beta_0[d] + beta_1[d]*urban[i] + beta_2[d]*living_children[i] + beta_3[d]*age_mean[i];
    y_rep[i] = bernoulli_logit_rng(p);
  }
}

