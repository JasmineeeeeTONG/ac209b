//solution_5b.stan

data {
      int nteams; // number of teams
      int ngames; // number of games
      int<lower=1, upper=nteams> team1[ngames]; // team 1 ID (1, ..., 30)
      int<lower=1, upper=nteams> team2[ngames]; // team 2 ID (1, ..., 30)
      int<lower=0, upper=1> team1_win[ngames]; // binary variable of whether home team won
}

parameters {
      real<lower=0> sigma; // talent level variation hyperparameter
      vector[nteams] lambda; // team talent levels in log space
}

transformed parameters {
     vector[nteams] exp_lambda; // team talent levels back in real space
     exp_lambda = exp(lambda);
}

model {
     sigma ~ uniform(0,50);
     lambda ~ normal(0, sigma);
     for (i in 1:ngames) {
        team1_win[i] ~ bernoulli(exp_lambda[team1[i]] / (exp_lambda[team1[i]] + exp_lambda[team2[i]]));
     }
}
