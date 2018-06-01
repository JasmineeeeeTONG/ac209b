data {
  int N; // Number of observations
  int J; // Number of districts
  int district[N];
  real urban[N];
  real living_children[N];
  real age_mean[N];
  int Y[N]; // Bernoulli Response: {0, 1} array
  int N_test; // Number of observations in test data
  int district_test[N_test];
  real urban_test[N_test];
  real living_children_test[N_test];
  real age_mean_test[N_test];
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
    real lin_p = beta_0[d] + beta_1[d]*urban[i] + beta_2[d]*living_children[i] + beta_3[d]*age_mean[i];
    Y[i] ~ bernoulli_logit(lin_p);
  }
}

generated quantities {
  int y_pred[N_test];         // Draws from posterior predictive dist
  for (i in 1:N_test) {
    int d = district_test[i];
    real lin_pred = beta_0[d] + beta_1[d]*urban_test[i] + beta_2[d]*living_children_test[i] + beta_3[d]*age_mean_test[i];
    y_pred[i] = bernoulli_logit_rng(lin_pred);
  }
}

