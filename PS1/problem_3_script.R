suppressMessages(library(ggplot2))
suppressMessages(library(dplyr))

# Part A ------------------------------------------------------------------
N = 50 # sample size
SIGMA = .2 # nuisance sd
YHAT = 1.3 # point to evaluate
S = 1000 # number of sims
B = 10000 # number of bootstraps

generate_data <- function(n=N, sigma=SIGMA, seed=11){
  #' Function to sample an X ~ U(0, 1), and, on top of that x, compute  Y ~ N(1 + x^2, sigma^2)
  #' @param n: int, sample size. 
  #' @param sigma: numeric, standard deviation of epsilon
  #' @param seed: int, sample seed for reproducability and/or recovery of "first" sample.
  #' @return list[x=numeric, y=numeric]: A paired list of x, y
  
  set.seed(seed)
  x = runif(n, 0, 1)
  y = rnorm(n, 1 + x^2, sigma)
  list(x=x, y=y)
}


fit_and_solve <- function(x, y, yhat, domain=c(0, 1)){
  #' Fits the quadratic regression specified in the problem, and then predicts for a particular yhat
  #' @param x: numeric[n], a vector of (sampled) x-values that are the covariates of the regression.
  #' @param y: numeric[n], a vector of (sampled) y-values that are the targets of the regression.
  #' @param yhat: numeric[k], yhats to be predicted (in reverse), in accordance with the problem
  #' @param domain: numeric[2], the intended domain for xhat. Results outside this range will be ignored
  #' @return numeric : the predicted value of xhat
  
  # fit the model
  mod = lm(y ~ 1 + x + I(x^2))
  # extract coefficients
  beta_fit = as.numeric(mod$coefficients)
  # cover the imaginary cases
  sqrt_term = beta_fit[2]^2 - 4 * (beta_fit[1] - yhat) * beta_fit[3]
  sqrt_term = max(sqrt_term, 0)
  # solve for roots
  x0_hat = c((-beta_fit[2] + sqrt(sqrt_term))/(2 * beta_fit[3]),
             (-beta_fit[2] - sqrt(sqrt_term))/(2 * beta_fit[3]))
  if (length(x0_hat[x0_hat >= domain[1] & x0_hat <= domain[2]]) == 2){
    warning(paste0(length(x0_hat[x0_hat >= domain[1] & x0_hat <= domain[2]]), 
                   " valid roots detected -- sampling one of them at random"))
    return(sample(x0_hat[x0_hat >= domain[1] & x0_hat <= domain[2]], 1))
  }
  # restrict root to appropriate range
  return(x0_hat[x0_hat >= domain[1] & x0_hat <= domain[2]])
}

# Part B ------------------------------------------------------------------

data = generate_data()
xhat_b = fit_and_solve(x=data$x, y=data$y, yhat=YHAT)

# Part C ------------------------------------------------------------------
xhats_c = sapply(1:S, function(i){data = generate_data(seed=i+1); fit_and_solve(x=data$x, y=data$y, yhat=YHAT)})

# Part D ------------------------------------------------------------------

xhats_d = sapply(1:B, function(i){
  slc = sample(1:N, N, replace=T); # draw the sample
  fit_and_solve(x=data$x[slc], y=data$y[slc], yhat=YHAT) # perform the procedure
}) %>% 
  unlist()

# Part E ------------------------------------------------------------------

sd_c = sd(xhats_c, na.rm=T)
sd_d = sd(xhats_d, na.rm=T)
ggplot(data.frame(xhat = c(xhats_c, xhats_d),
                          method = c(rep("Simulated", length(xhats_c)),
                                     rep("Bootstrapped", length(xhats_d))
                          )),
               aes(x=xhat,
                   color=method)) +
  geom_density() +
  xlim(c(.25, .75))

# Part F ------------------------------------------------------------------

# method 1: straightaway:
# 1.) take B bootstraps and compute quantiles
ci_method_1 = quantile(xhats_d, c(.05, .95), na.rm=T) %>% suppressMessages()

# re-use xhats

# method 2: straightaway
# 1.) Take the distribution of bootstrap statistics as normal.
# 2.) Compute the confidence interval according to a normal distribution
ci_method_2 = qnorm(c(.05, .95), mean=mean(xhats_d), sd=sd(xhats_d))

# Plot the two methods (method 1 straight, method 2 dotted)
ggplot(data.frame(xhat = c(xhats_d), 
                  method = rep("Bootstrapped", 
                               length(xhats_d))),
       aes(x=xhat,
           color=method)) +
  geom_density() +
  geom_vline(xintercept=ci_method_1) +
  geom_vline(xintercept=ci_method_2, linetype="dotted") +
  xlim(c(.4, .7)) 

