data {
  int N; // # observations
  int M; // # features = 11
  int J; // Number of clusters = 6
  int clusters[N];
  real<lower=0> fixed_acidity[N];
  real<lower=0> volatile_acidity[N];
  real<lower=0> citric_acidity[N];
  real<lower=0> residual_suger[N];
  real<lower=0> chlorides[N];
  real<lower=0> free_sulfur_dioxide[N];
  real<lower=0> total_sulfur_dioxide[N];
  real<lower=0> density[N];
  real<lower=0> pH[N];
  real<lower=0> sulphates[N];
  real<lower=0> alcohol[N];
  int<lower=0> quality[N]; // response
  real mean_quality; // mean(quality)
}

parameters {
  real<lower=0> sigma;
  real<lower=0> sigma_0;
  real<lower=0> sigma_m[M];
  real beta_0[J];
  real beta_ij[M, J];
}

model {
  // Prior
  sigma ~ uniform(0, 100);
  sigma_0 ~ uniform(0, 100);
  for (i in 1:M) {
    sigma_m[i] ~ uniform(0, 100);
  }
  
  for (j in 1:J) {
    beta_0[j] ~ normal(mean_quality, sigma_0);
  }
  for (i in 1:M) {
    for (j in 1:J) {
      beta_ij[i, j] ~ normal(0, sigma_m[i]);
    }
  }
  
  // Likelihood
  for (n in 1:N) {
    int cluster = clusters[n];
    quality[n] ~ normal(beta_0[cluster] + 
    beta_ij[1, cluster] * fixed_acidity[n] + 
    beta_ij[2, cluster] * volatile_acidity[n] + 
    beta_ij[3, cluster] * citric_acidity[n] + 
    beta_ij[4, cluster] * residual_suger[n] + 
    beta_ij[5, cluster] * chlorides[n] + 
    beta_ij[6, cluster] * free_sulfur_dioxide[n] + 
    beta_ij[7, cluster] * total_sulfur_dioxide[n] + 
    beta_ij[8, cluster] * density[n] + 
    beta_ij[9, cluster] * pH[n] + 
    beta_ij[10, cluster] * sulphates[n] +
    beta_ij[11, cluster] * alcohol[n], sigma);
  }
}

generated quantities {
  real quality_rep[N]; // Draws from posterior predictive dist
  for (n in 1:N) {
    int cluster = clusters[n];
    quality_rep[n] = beta_0[cluster] + 
    beta_ij[1, cluster] * fixed_acidity[n] + 
    beta_ij[2, cluster] * volatile_acidity[n] + 
    beta_ij[3, cluster] * citric_acidity[n] + 
    beta_ij[4, cluster] * residual_suger[n] + 
    beta_ij[5, cluster] * chlorides[n] + 
    beta_ij[6, cluster] * free_sulfur_dioxide[n] + 
    beta_ij[7, cluster] * total_sulfur_dioxide[n] + 
    beta_ij[8, cluster] * density[n] + 
    beta_ij[9, cluster] * pH[n] + 
    beta_ij[10, cluster] * sulphates[n] +
    beta_ij[11, cluster] * alcohol[n];
  }
}
