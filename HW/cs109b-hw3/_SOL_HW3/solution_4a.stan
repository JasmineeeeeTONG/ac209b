//solution_4a.stan

data {
  int N; // number of observations
  int num_districts; // number of distinct districts
  int<lower=0, upper=1> contraceptive_use[N]; // response variable
  int<lower=1, upper=num_districts> district[N]; // district ID
  int<lower=0, upper=1> urban[N]; // binary variable of urban residence
  int<lower=0> living_children[N]; // number of living children
  real age_mean[N]; // mean age of women
}

parameters {
  vector[num_districts] a; // intercept for each district
  vector[num_districts] b1; // urban coef for each district
  vector[num_districts] b2; // age_mean coef for each district
  vector[num_districts] b3; // living_children coef for each district
  real mu_a; // mean of intercept prior
  real<lower=0,upper=100> sigma_a; // standard deviation of intercept prior
  real<lower=0,upper=100> sigma_b1; // standard deviaion of urban coef prior
  real<lower=0,upper=100> sigma_b2; // standard deviaion of age_mean coef prior
  real<lower=0,upper=100> sigma_b3; // standard deviaion of living_children coef prior
}

transformed parameters {
  vector[N] p_hat;
  for (i in 1:N)
           p_hat[i] = (a[district[i]] + b1[district[i]] * urban[i] + 
       b2[district[i]] * living_children[i] + 
       b3[district[i]] * age_mean[i]);
}

model {
  // priors
  mu_a ~ normal(0, 100);
  sigma_a ~ exponential(0.1);
  sigma_b1 ~ exponential(0.1);
  sigma_b2 ~ exponential(0.1);
  sigma_b3 ~ exponential(0.1);
  a ~ normal(mu_a, sigma_a);
  b1 ~ normal(0, sigma_b1);
  b2 ~ normal(0, sigma_b2);
  b3 ~ normal(0, sigma_b3);
  
  //likelihood
   contraceptive_use ~ bernoulli_logit(p_hat);
}
