suppressMessages(library(dplyr))
suppressMessages(library(ggplot2))
suppressMessages(library(mvtnorm))
suppressMessages(library(class))
suppressMessages(library(rlist))

PI = .5
K = 8
W = rep(1/8, K)
SIGMA = .5 * diag(2)

N_TRAIN = 300
N_TEST = 20000

# Part A ------------------------------------------------------------------

# generate mu
# get with key j + 1; then get row k by slicing
set.seed(11)
MU = list(
  rmvnorm(8, c(0, 1), diag(2)),
  rmvnorm(8, c(1, 0), diag(2))
)


# PART B ------------------------------------------------------------------
generate_mixed_normals <- function(n, centroids=MU, omega=W, sigma2=SIGMA^2, pi=PI, seed=25){
  #' Generates the density in 1a, pursuant to problem instructions
  #' @param n: int, size of generated dataset
  #' @param centroids: list[matrix[K, 2], matrix[K, 2]]. A list of the centroids
  #' @param omega: numeric[K] mixing weights
  #' @param sigma2: matrix[2, 2], var-cov matrix for MVN
  #' @param pi: numeric, class priors
  #' @param seed: integer, seed for reproducability
  #' @return : list[matrix[n, 2], numeric[n]], a list of x, y values
  
  set.seed(seed)
  pos_class = rbinom(n, 1, pi)
  
  # sample from omega, THEN sample from MVN
  # generate from positive class
  set.seed(220)
  pos_x = lapply(sample(1:length(omega), sum(pos_class), prob=omega, replace=T), 
                 function(k) rmvnorm(1, centroids[[2]][k, ], sigma2)) %>%
    do.call("rbind", .)
  # generate from negative class
  neg_x = lapply(sample(1:length(omega), length(pos_class) - sum(pos_class), prob=omega, replace=T), 
                 function(k) rmvnorm(1, centroids[[1]][k, ], sigma2)) %>%
    do.call("rbind", .)
  
  list(
    x=rbind(neg_x, pos_x), 
    y=c(rep(0, nrow(neg_x)), rep(1, nrow(pos_x)))
  )
}

get_mixed_class_density <- function(x, mu, sigma=SIGMA^2, omega=W){
  #' Gets the mixture/mixed density for a single class, i.e. j=0 OR j=1
  #' @param x: matrix[n, 2], a matrix of x-values to take the density of
  #' @param sigma: matrix[2, 2], variance-covariance matrix of distribution
  #' @param omega: numeric[K], mixing weights
  #' @return : numeric[n], conditional mixed densities for the class
  
  lapply(1:nrow(mu), 
                   function(k) dmvnorm(x, 
                                       mu[k, ], 
                                       sigma) * omega[k]
  ) %>%
    base::Reduce("+", .)
}

# PART D ------------------------------------------------------------------
compute_bayes_classifier <- function(x, centroids, sigma=SIGMA^2, omega=W, pi=PI, decision_boundary=.5, response){
  #' Computes the bayes classifier at a particular x
  #' @param x: matrix[n, 2], a matrix of x-values to evaluate
  #' @param centroids: list[matrix[K, 2], matrix[K, 2]]. A list of the centroids
  #' @param omega: numeric[K] mixing weights
  #' @param sigma: matrix[2, 2], var-cov matrix for MVN
  #' @param pi: numeric, class priors
  #' @param decision_boundary: numeric, cutoff for 1/0 classification
  #' @return : numeric[nrow(x)], the Bayes classifications of each entry in x

  dens_pos = get_mixed_class_density(x, mu=centroids[[2]], sigma, omega)
  dens_neg = get_mixed_class_density(x, mu=centroids[[1]], sigma, omega)
  
  # doing it in full in case pi ever changes
  if (response == 'prob'){
    return(
      (dens_pos*pi)/(dens_pos*pi + dens_neg*(1-pi))
      )
  }else{
    return(
      (dens_pos*pi)/(dens_pos*pi + dens_neg*(1-pi)) >= decision_boundary
    )
  }
}

# PART E ------------------------------------------------------------------
# linear classifier
fit_linear_classifier <- function(x, y){
  #' Fits the linear classifier
  #' @param x: matrix[n, 2], a matrix of features
  #' @param y: numeric[n], a vector of binary y-values
  #' @return : stats::lm(), a linear model
  # R really wants it to be a df I guess
  df = data.frame(x)
  df$y = y
  lm(y ~ ., df)
}

# linear model prediction
predict_linear_classifier <- function(mod, x, decision_boundary=.5){
  #' Predicts for a fitted linear classifier
  #' @param mod: stats::lm(), a fitted linear model
  #' @param x: matrix[n, 2], a matrix of x-values to predict
  #' @param decision_boundary: numeric, the 1/0 decision cutoff
  #' @return : numeric, predicted y classes

  yhat = predict(mod, newdata=data.frame(x))
  as.numeric(yhat >= decision_boundary)
}

# evaluation function
train_test_model_suite <- function(the_data, 
                                   kvec=seq(1, 15, 2), 
                                   params=list(
                                     centroids=MU,
                                     omega=W,
                                     pi=PI,
                                     sigma=SIGMA^2
                                   )){
  # Bayes classifier
  bayes_test_class = compute_bayes_classifier(the_data$xtest[, c(1,2)], centroids=params$centroids, sigma=params$sigma, omega=params$omega, pi=params$pi,
                                              response='class')
  bayes_test_acc = mean(bayes_test_class == the_data$ytest)
  
  # Linear classifier
  clf_linear = fit_linear_classifier(the_data$xtrain, the_data$ytrain)
  yhat_linear = predict_linear_classifier(clf_linear, the_data$xtest)
  linear_test_acc = mean(yhat_linear == the_data$ytest)
  acc_list = c(bayes_test_acc, linear_test_acc) # list of accuracy measurements
  
  # KNN classifiers
  for (k in kvec) {
    yhat_KNN = knn(train=the_data$xtrain, test=the_data$xtest, cl=as.factor(the_data$ytrain), k=k, l=0, prob=FALSE, use.all=TRUE)
    KNN_test_acc = mean(yhat_KNN == the_data$ytest)
    acc_list = c(acc_list, KNN_test_acc)
  }
  return(acc_list)
}

# PART F ------------------------------------------------------------------
# noisy wrapper around test suite
noisy_evaluation <- function(kvec, the_data, noise, sigma.noise) {
  # Add noise columns
  xtrain.noise <- replicate(noise, rnorm(nrow(the_data$xtrain), mean=0, sd=sigma.noise))
  xtest.noise <- replicate(noise, rnorm(nrow(the_data$xtest), mean=0, sd=sigma.noise))
  the_data$xtrain <- cbind(the_data$xtrain, xtrain.noise)
  the_data$xtest <- cbind(the_data$xtest, xtest.noise)
  train_test_model_suite(the_data, kvec)
}

# main portion ------------------------------------------------------------

# PART C ------------------------------------------------------------------
trainset = generate_mixed_normals(n=N_TRAIN)
xtrain = trainset$x; ytrain = trainset$y
testset = generate_mixed_normals(n=N_TEST)
xtest = testset$x; ytest = testset$y

the_data = list(
  xtrain=xtrain,
  ytrain=ytrain,
  xtest=xtest,
  ytest=ytest
)

mm = data.frame(the_data$xtest)
mm$yhat = compute_bayes_classifier(mm, centroids=MU, sigma=SIGMA^2, omega=W, pi=PI, response='class')
mm$y = the_data$ytest

# PART E ------------------------------------------------------------------
kvec = seq(1, 15, 2)
train_test_model_suite(the_data, kvec)

# PART F ------------------------------------------------------------------
# adding noise parameters
sigma.noise = 1
noise.cols = 1:10

noisy.result <- sapply(noise.cols, function(n_cols) {
  return(noisy_evaluation(kvec, the_data, n_cols, sigma.noise))
})

# Plot the effects of adding the noise parameters on the accuracy
res <- matrix(noisy.result, nrow=10)
plot(t(res)[,1], type="l", xlab="# noise parameters", ylab="accuracy rate", ylim=c(0.5, 1.0))
lines(t(res)[,2], col="red")
for (i in 3:10) {
  lines(t(res)[,i], col="blue")
}
legend("bottomleft", 
       legend=c("Bayes", "Linear", "KNN"), 
       fill=c("black", "red", "blue"))

# Some basic plots/visualization ------------------------------------------

x_grid = expand.grid(seq(-3, 3, .01), seq(-3, 3, .01))
bayes_x_grid = compute_bayes_classifier(x_grid, centroids=MU, sigma=SIGMA^2, omega=W, pi=PI, response='class') %>% as.numeric()
x_grid = data.frame(x_grid) %>%
  `colnames<-`(c("X1", "X2"))
x_grid$y = bayes_x_grid

ggplot(x_grid, aes(x=X1, y=X2, color=as.factor(y)), alpha=.01) + 
  geom_point(size=.5) +
  scale_color_discrete(name='y') + 
  geom_point(data = bind_rows(data.frame(MU[[1]]) %>% mutate(y=0), data.frame(MU[[2]]) %>% mutate(y=1)),
             mapping=aes(x=X1, y=X2), size= 5, colour='black') +
  geom_point(data = bind_rows(data.frame(MU[[1]]) %>% mutate(y=0), data.frame(MU[[2]]) %>% mutate(y=1)),
             mapping=aes(x=X1, y=X2, color = as.factor(y)), size= 4) + 
  labs(title = 'Bayes Decision Boundary', x='x1', y='x2', subtitle = 'Large points= centroids')

# the training data
ggplot(data.frame(cbind(the_data$xtrain, the_data$ytrain)) %>% `colnames<-`(c("x1", "x2", "y")),
       aes(x=x1, y=x2, color=as.factor(y))) +
  geom_point(alpha=.4) +
  scale_color_discrete(name='y') + 
  geom_point(data = bind_rows(data.frame(MU[[1]]) %>% mutate(y=0), data.frame(MU[[2]]) %>% mutate(y=1)),
             mapping=aes(x=X1, y=X2), size= 5, colour='black') +
  geom_point(data = bind_rows(data.frame(MU[[1]]) %>% mutate(y=0), data.frame(MU[[2]]) %>% mutate(y=1)),
             mapping=aes(x=X1, y=X2, color = as.factor(y)), size= 4) + 
  labs(title = 'Distribution of Training x1, x2, y', subtitle='Large points = centroids')


# the test data
ggplot(data.frame(cbind(the_data$xtest, the_data$ytest)) %>% `colnames<-`(c("x1", "x2", "y")),
       aes(x=x1, y=x2, color=as.factor(y))) +
  geom_point(alpha=.4) +
  scale_color_discrete(name='y') + 
  geom_point(data = bind_rows(data.frame(MU[[1]]) %>% mutate(y=0), data.frame(MU[[2]]) %>% mutate(y=1)),
             mapping=aes(x=X1, y=X2), size= 5, colour='black') +
  geom_point(data = bind_rows(data.frame(MU[[1]]) %>% mutate(y=0), data.frame(MU[[2]]) %>% mutate(y=1)),
             mapping=aes(x=X1, y=X2, color = as.factor(y)), size= 4) + 
  labs(title = 'Distribution of Test x1, x2, y', subtitle='Large points = centroids')


