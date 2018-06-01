//solution_3b.stan

data{
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
     vector[3] b; // fixed slopes
    real<lower=0,upper=100> sigma_a; // standard deviation of intercept prior
     real mu_a; // mean of intercept prior
   }

transformed parameters {
     vector[N] p_hat;
     for (i in 1:N)
       p_hat[i] = (b[1] * urban[i] + b[2] * living_children[i] + b[3] * age_mean[i] +
                   a[district[i]]);
}
model {
     
     // priors
     mu_a ~ normal(0, 100);
     sigma_a ~ exponential(0.1);
     a ~ normal(mu_a, sigma_a);
     b ~ normal(0, 100);
     
     // likelihood
     contraceptive_use ~ bernoulli_logit(p_hat);
   }
