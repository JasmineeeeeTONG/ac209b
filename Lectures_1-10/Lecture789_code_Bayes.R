# analyses for Bayesian notes

x = (0:100)/100

# uniform prior
postscript(file="uniform-prior.ps",hori=T)
plot(x, dunif(x), type='l', lwd=3, col="red",
main=expression(paste("Uniform prior distribution for ",theta)),
xlab=expression(theta),
ylab=expression(p(theta)))

# likelihood
postscript(file="likelihood-bern.ps",hori=T)
plot(x, x*(1-x)^4, type='l', lwd=3, col="red",
main="Likelihood function for 1 death, 4 survivals",
xlab=expression(theta),
ylab=expression(paste(L(theta* "|" *y[1],y[2],y[3],y[4],y[5]))))

# posterior
postscript(file="beta.ps",hori=T)
plot(x, dbeta(x,2,5), type='l', lwd=3, col="red",
main="Be(2,5)",
xlab=expression(theta),
ylab=expression(p(theta*"|"*y)))

# posterior with intervals
postscript(file='beta95.ps',hori=T)
plot(x, dbeta(x,2,5), type='l', lwd=3, col="red",
main="Posterior distribution, with 95% intervals",
xlab=expression(theta),
ylab=expression(p(theta*"|"*y)))
x025 = qbeta(0.025, 2, 5)
x975 = qbeta(0.975, 2, 5)
segments(x025, 0, x025, dbeta(x025,2,5), lty=5, lwd=3, col="blue")
segments(x975, 0, x975, dbeta(x975,2,5), lty=5, lwd=3, col="blue")
fr025 = 0.011
fr975 = 0.701
segments(fr025, 0, fr025, dbeta(fr025,2,5), lty=5, lwd=3, col="darkgreen")
segments(fr975, 0, fr975, dbeta(fr975,2,5), lty=5, lwd=3, col="darkgreen")
legend(0.6,2.0, legend=c("95% central posterior interval",
                         "95% confidence interval"),
  col=c("blue","darkgreen"),lty=c(5,5),lwd=c(3,3))

postscript(file="theta-seq.ps",hor=T)
par(mfrow=c(2,3))
plot(x, dbeta(x,1,1), type='l', lwd=3, col="red",
main=expression(paste("Uniform prior distribution for ",theta)),
xlab=expression(theta),
ylab=expression(paste(p(theta))))
plot(x, dbeta(x,1,2), type='l', lwd=3, col="red",
main=expression(y[1]==0),
xlab=expression(theta),
ylab=expression(paste(p(theta* "|" *y[1]))))
plot(x, dbeta(x,1,3), type='l', lwd=3, col="red",
main=expression(paste(y[1]==0,", ",y[2]==0)),
xlab=expression(theta),
ylab=expression(paste(p(theta* "|" *y[1],y[2]))))
plot(x, dbeta(x,1,4), type='l', lwd=3, col="red",
main=expression(paste(y[1]==0,", ",y[2]==0,", ",y[3]==0)),
xlab=expression(theta),
ylab=expression(paste(p(theta* "|" *y[1],y[2],y[3]))))
plot(x, dbeta(x,2,4), type='l', lwd=3, col="red",
main=expression(paste(y[1]==0,", ",y[2]==0,", ",y[3]==0,
", ",y[4]==1)),
xlab=expression(theta),
ylab=expression(paste(p(theta* "|" *y[1],y[2],y[3],y[4]))))
plot(x, dbeta(x,2,5), type='l', lwd=3, col="red",
main=expression(paste(y[1]==0,", ",y[2]==0,", ",y[3]==0,
", ",y[4]==1,", ",y[5]==0)),
xlab=expression(theta),
ylab=expression(paste(p(theta* "|" *y[1],y[2],y[3],y[4],y[5]))))

# next plot didn't make it into the lecture notes!

y0 = dbeta(x,1,1)
y1 = dbeta(x,1,2)
y2 = dbeta(x,1,3)
y3 = dbeta(x,2,3)
y4 = dbeta(x,2,4)
y5 = dbeta(x,2,5)
yylim=range(c(y0,y1,y2,y3,y4,y5))
postscript(file='theta-seq2.ps',hori=T)
par(mfrow=c(2,3))

plot(x,y0,type='n',
main="Sequential updating", xlab=expression(theta),ylim=yylim,
ylab=expression(p(theta)))
lines(x,y0,lwd=3,col="green")

plot(x,y0,type='n',
main="Sequential updating", xlab=expression(theta),ylim=yylim,
ylab=expression(p(theta)))
lines(x,y0,lwd=3,col="green")
lines(x,y1,lwd=3,col="purple")

plot(x,y0,type='n',
main="Sequential updating", xlab=expression(theta),ylim=yylim,
ylab=expression(p(theta)))
lines(x,y0,lwd=3,col="green")
lines(x,y1,lwd=3,col="purple")
lines(x,y2,lwd=3,col="orange")

plot(x,y0,type='n',
main="Sequential updating", xlab=expression(theta),ylim=yylim,
ylab=expression(p(theta)))
lines(x,y0,lwd=3,col="green")
lines(x,y1,lwd=3,col="purple")
lines(x,y2,lwd=3,col="orange")
lines(x,y3,lwd=3,col="seagreen")

plot(x,y0,type='n',
main="Sequential updating", xlab=expression(theta),ylim=yylim,
ylab=expression(p(theta)))
lines(x,y0,lwd=3,col="green")
lines(x,y1,lwd=3,col="purple")
lines(x,y2,lwd=3,col="orange")
lines(x,y3,lwd=3,col="seagreen")
lines(x,y4,lwd=3,col="darkturquoise")

plot(x,y0,type='n',
main="Sequential updating", xlab=expression(theta),ylim=yylim,
ylab=expression(p(theta)))
lines(x,y0,lwd=3,col="green")
lines(x,y1,lwd=3,col="purple")
lines(x,y2,lwd=3,col="orange")
lines(x,y3,lwd=3,col="seagreen")
lines(x,y4,lwd=3,col="darkturquoise")
lines(x,y5,lwd=3,col="red")

# simulate from Be(2,5)
theta = rbeta(10000, 2, 5)  # generate 10,000 values from Be(2,5)
head(theta)  # first six simulated values
mean(theta)  # sample mean
quantile(theta,probs=c(0.025,0.975))  # 2.5% and 97.5% of empirical distn

# simulate y from posterior predictive distribution
theta = rbeta(10000, 2, 5)  # generate 10,000 values from Be(2,5)
head(theta)
y = rbinom(10000, 1, theta) # generate 10,000 binary values
                            # with different probabilities
head(y)
table(y)/10000   # frequency of 0 and 1

# Bayesian logistic regression - Beetles survival example
library("rstan")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

beetles.dat = list(
x = c(1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839),
n = c(59, 60, 62, 56, 63, 59, 62, 60),
y = c(6, 13, 18, 28, 52, 53, 61, 60),
N = 8)

beetles.fit = stan(file="beetles.stan", data=beetles.dat,
  iter = 3000, warmup=500, chains = 4)
print(beetles.fit)
postscript(file="beetles-trace.ps",hori=T)
trace= traceplot(beetles.fit,pars=c("alpha_star","beta"),inc_warmup=T)
trace + scale_color_discrete() + theme(legend.position="top")
dev.off()

# Sleep study
library(lme4) # to get sleepstudy data

require(lattice)
postscript(file="reaction-xy.ps",hori=T)
xyplot(Reaction ~ Days | Subject, sleepstudy, type = c("g","p","r"),
index = function(x,y) coef(lm(y ~ x))[1],
main="Sleep deprivation - reaction times with least-squares fits",
col="red",
pch=19,
xlab = "Days of sleep deprivation",
ylab = "Average reaction time (ms)", aspect = "xy")

# run hierarchical model
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

mysleep = sleepstudy # local copy
# trick to create IDs from 1 to 18 based on Subject variable
mysleep$ID = as.numeric(factor(mysleep$Subject))
sleep.dat = list(
   N = nrow(mysleep),
   N_ID = max(mysleep$ID),
   y = mysleep$Reaction,
   x = mysleep$Days,
   id = mysleep$ID
   )
sleep.fit = stan(file="sleep.stan", data=sleep.dat,
  iter = 5000, warmup=2000, chains = 3)
linpred.names = paste("linpred[",1:sleep.dat$N,"]",sep='')
mysleep$Reaction.pred = summary(sleep.fit)$summary[linpred.names,1]

library(latticeExtra)

postscript(file="reaction-pred-obs.ps",hori=T)
plot(mysleep$Reaction,mysleep$Reaction.pred,pch=19,
main = "Observed and fitted reaction times
from Bayesian hierarchical linear model",
xlab = "Observed reaction time (ms)",
ylab = "Predicted reaction time (ms)" )
abline(0,1,lwd=2, col="blue")

# add predicted Y to xyplot
postscript(file="reaction-xy-1.ps",hori=T)
xyplot(Reaction ~ Days | Subject,
   mysleep, type = c("g","p","r"),
index = function(x,y) coef(lm(y ~ x))[1],
col="red",
lwd=2,
pch=19,
subscript=T,
main = "Sleep deprivation - pooling data across all subjects",
xlab = "Days of sleep deprivation",
ylab = "Average reaction time (ms)", aspect = "xy")  +
xyplot(Reaction~Days,
  index = function(x,y) coef(lm(y ~ x))[1],
  type="r", aspect="xy",
  col="black",lwd=2,
  data=mysleep)

# add predicted Y to xyplot
postscript(file="reaction-xy-2.ps",hori=T)
xyplot(Reaction ~ Days | Subject,
   mysleep, type = c("g","p","r"),
index = function(x,y) coef(lm(y ~ x))[1],
col="red",
lwd=2,
pch=19,
subscript=T,
main = "Sleep deprivation - hierarchical model fit",
xlab = "Days of sleep deprivation",
ylab = "Average reaction time (ms)", aspect = "xy")  +
xyplot(Reaction.pred~Days | Subject,
  index = function(x,y) coef(lm(y ~ x))[1],
  type="r", aspect="xy",
  col='purple',lwd=2, data=mysleep) +
xyplot(Reaction~Days,
  index = function(x,y) coef(lm(y ~ x))[1],
  type="r", aspect="xy",
  col="black",lwd=2,
  data=mysleep)

# Harry Potter LDA analysis
library(harrypotter) #harry potter books 1-7
library(topicmodels) #topic modeling functions
library(stringr) #common string functions
library(tidytext) #tidy text analysis
library(tidyverse) #data manipulation and visualization
library(scales) #used for percent scale on confusion table

## Source topicmodels2LDAvis & optimal_k functions
invisible(lapply(
file.path(
"https://raw.githubusercontent.com/trinker/topicmodels_learning/master/functions", 
c("topicmodels2LDAvis.R", "optimal_k.R")
),
devtools::source_url
))

ChamSec <- rowid_to_column(tibble(text = chamber_of_secrets, title = "Chamber Secrets"), var = "chapter")
DeathHall <- rowid_to_column(tibble(text = deathly_hallows, title = "Deathly Hallows"), var = "chapter")
GobFire <- rowid_to_column(tibble(text = goblet_of_fire, title = "Goblet Fire"), var = "chapter")
HalfBlood <- rowid_to_column(tibble(text = half_blood_prince, title = "Half Blood Prince"), var = "chapter")
OrderPhoenix <- rowid_to_column(tibble(text = order_of_the_phoenix, title = "Order Phoenix"), var = "chapter")
PhilStone <- rowid_to_column(tibble(text = philosophers_stone, title = "Philosophers Stone"), var = "chapter")
PrisAzkaban <- rowid_to_column(tibble(text = prisoner_of_azkaban, title = "Prisoner Azkaban"), var = "chapter")

by_chapter <- rbind(ChamSec, DeathHall, GobFire, 
  HalfBlood, OrderPhoenix, PhilStone, PrisAzkaban)
by_chapter <- by_chapter %>%
  unite(document, c("title", "chapter"), sep = "_", remove = TRUE)

by_chapter_word <- by_chapter %>%
  unnest_tokens(word, text)

# find document-word counts
word_counts <- by_chapter_word %>%
  anti_join(stop_words) %>%
  count(document, word, sort = TRUE) %>%
  ungroup()

stopwords <- add_row(stop_words, 
word = c("harry","hermione", "ron"), lexicon = c("SMART", "SMART", "SMART"))

# find document-word counts
word_counts <- by_chapter_word %>%
  anti_join(stopwords) %>%
  count(document, word, sort = TRUE) %>%
  ungroup()

chapters_dtm <- word_counts %>%
  cast_dtm(document, word, n)

# LDA
control <- list(burnin = 500, iter = 1000, keep = 100, seed = 2500)
opt.k = optimal_k(chapters_dtm, max.k=30, control=control)
# optimum comes out to 22
postscript(file="lda-optk.ps",hori=T)
opt.k
dev.off()

chapters_lda = LDA(chapters_dtm, k = as.numeric(opt.k), 
                   method="Gibbs", control=control)
lda_inf = posterior(chapters_lda)
topics.hp = topics(chapters_lda,1)
terms.hp = terms(chapters_lda, 10)
print(terms.hp[,1:5])

chapter_topics <- tidy(chapters_lda, matrix = "beta")
top_n(chapter_topics, 10)
top_terms <- chapter_topics %>%
  group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

postscript(file="lda-beta.ps",hori=T)
top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()
dev.off()

chapters_gamma <- tidy(chapters_lda, matrix = "gamma")

chapters_gamma <- chapters_gamma %>%
  separate(document, c("title", "chapter"), sep = "_", convert = TRUE)

top_n(chapters_gamma, 10)

postscript(file="lda-gamma.ps",hori=T)
chapters_gamma %>%
  mutate(title = reorder(title, gamma * topic)) %>%
  ggplot(aes(factor(topic), gamma)) +
  geom_boxplot() +
  facet_wrap(~ title)
dev.off()
