library(dplyr)

# TODO: scaler function

lm_step_construct <- function(X, y){
  #' Constructor function for lm_step object
  #' @param X: matrix. A matrix of features to train on.
  #' @param y: numeric. A vector of targets to train on
  #' @return : lm_step. a constructed lm_step object. 

  # TODO: integrity checks for inputs
  Z = matrix(rep(0, nrow(X) * (ncol(X)+1)), ncol=ncol(X)+1)
  Z[, 1] = 1
  lm_init = list(
    n=nrow(X), # number of samples
    p=ncol(X), # max no. predictors
    k=0, # no features chosen yet 
    X=X, # the data
    X_sel=c(), # indices of columns in X selected
    X_nonsel=1:ncol(X), # indices of columns in X not selected
    Z=Z, # decomposition into z_0, ...; updated througout selection
    y=y, # targets
    beta_step=matrix(rep(0, ncol(X)^2), ncol=ncol(X)), # log of betas
    is_fit=F, # for checks
    j=1 # current predictor number
  )
  attr(lm_init, "class") = "lm_step"
  lm_init
}


get_zj <- function(lm_step, x_idx){
  #' Given z0, ..., z_{j-1}, this function computes z_j
  #' @param lm_step: lm_step. An lm_step model
  #' @param x_idx: integer. The index of the column in X to which z_j is computed with respect.
  #' @return : numeric. A vector of z_j. 

  # compute dot products to get individual gammas
  gamma_vec = sapply(1:lm_step$j, function(ll)
    sum(lm_step$Z[, ll] * lm_step$X[, x_idx])/sum(lm_step$Z[, ll] * lm_step$Z[, ll])
    )
  # apply these gammas as coefficients to z
  
  z_j = lm_step$X[, x_idx] - lapply(1:length(gamma_vec), function(ll)
    gamma_vec[ll]*lm_step$Z[, ll]
    ) %>%
    do.call("rbind", .) %>%
    colSums()
  z_j
}

advance_zj <- function(lm_step, x_idx){
  #' Adds a z_j corresponding to one of the X columns to the running z's
  #' @param lm_step: lm_step. An lm_step model
  #' @param x_idx: integer. The index of the column in X to which z_j is computed with respect.
  #' @return : lm_step. The model, with z having been updated/incremented by one. 
  z_j = get_zj(lm_step, x_idx)
  lm_step$j = lm_step$j + 1
  lm_step$Z[, lm_step$j] = z_j
  # TODO: write tests for orthogonality of new z_j
  lm_step
}

solve_beta_j <- function(lm_step, x_idx){
  #' Computes B_j for "next" feature j. 
  #' @param lm_step: lm_step. An lm_step model
  #' @param x_idx: integer. The index of the column in X, for which beta is being computed
  #' @return : numeric. An estimate of beta_hat_j
  
  sum(lm_step$Z[, lm_step$j] * lm_step$y)/sum(lm_step$Z[, lm_step$j] * lm_step$Z[, lm_step$j])
}


fit_ols_successive <- function(lm_step, x_idxs){
  #' Fits least squares coefficients via successive orthogonalization. Data assumed scaled, does not touch intercept.
  #' @param lm_step: lm_step. An lm_step model
  #' @param x_idxs: numeric[p]. Indices of features. Coefficients are returned in order respective to this input
  #' @return : numeric[p]. OLS Coefficients. Index corresponds to that passed in x_idxs.
  
  # make a copy, so as not to taint values of the underlying matrix
  beta_vec = sapply(x_idxs, function(ll){
    # make a copy so Z's don't get tangled
    lm_step_copy = lm_step;
    # do the successive orthogonalization
    for (i in c(setdiff(x_idxs, ll), ll)){
      lm_step_copy = advance_zj(lm_step=lm_step_copy, x_idx=i);
    };
    # regress y ~ z_jto get beta_j
    beta_prop = solve_beta_j(lm_step=lm_step_copy, x_idx=ll);
    # return
    beta_prop;
  })
  beta_vec
}

test_proposed_beta <- function(lm_step, beta_vec, which_beta, x_idxs){
  #' Tests a proposed beta via T-Test
  #' @param lm_step: lm_step. An lm_step object
  #' @param beta_vec: numeric. A vector of coefficients
  #' @param which_beta: integer. The index of beta_vec that is the coefficient to test
  #' @param x_idxs: numeric[length(beta_vec)]. The indices in X corresponding to beta_vec. 
  # TODO: test to ensure which_beta <= length(beta_vec)
  # which of the betas are we testing
  beta_j = beta_vec[which_beta]
  # extract appropriate features from X
  X_subset = lm_step$X[, x_idxs]
  # compute it
  XTX = solve(t(X_subset) %*% X_subset)
  # the new row will be last, since x_idx came last
  vj = XTX[which_beta, which_beta]
  # compute it
  # if this is our first variable
  if (is.null(nrow(X_subset))){
    yhat = mean(lm_step$y) + X_subset * beta_vec
  }else{
    # otherwise, append new variable: note J has already been advanced, so we must take an extra step back to get
    # variables already saved
    yhat = mean(lm_step$y) + X_subset %*% beta_vec
  }
  # record df
  deg_freedom = lm_step$n - length(beta_vec) - 1
  # compute sigma hat sq
  sigma_hat_sq = sum(t(lm_step$y - yhat) %*% (lm_step$y - yhat))/deg_freedom
  # put it all together
  test_statistic = beta_j/sqrt(sigma_hat_sq * vj)
  # compute pval: t-distribution
  pval = pt(-abs(test_statistic), df=deg_freedom) + 
    1 - pt(abs(test_statistic), df=deg_freedom)
  # TODO: standard normal toggle
  t(c(pval, test_statistic))
}


evaluate_single_step <- function(lm_step){
  #' Considers new features to add to a model on a given iteration.
  #' @param lm_step: lm_step. The model, having selected some number or no features already
  #' @return : numeric. A vector containing: the index of the best feature to add; 
  #' the p-value for the hypothesis test for that coefficient; and hypothesis test for the test of that coefficient. 
  
  # which of the columns haven't we used up
  eligible_col_idx = lm_step$X_nonsel
  results = sapply(eligible_col_idx, function(ll){
    # betas for this permutation of columns
    # THE FIRST COLUMN IS THE NEW ONE!  
    beta_iter = fit_ols_successive(lm_step, c(ll, lm_step$X_sel))
    test_proposed_beta(lm_step, beta_vec=beta_iter, which_beta=1, x_idxs=c(ll, lm_step$X_sel))
  })
  
  winning_feature_idx = which.min(results[1, ])
  # return index of winning feature, beta, and corresponding p-value
  c(eligible_col_idx[winning_feature_idx], 
    results[, winning_feature_idx])
}


select_stepwise <- function(lm_step, alpha=.05){
  #' Helper to perform the stepwise selection
  #' @param lm_step: lm_step. An unfitted lm_step model, soon to be fitted.
  #' @param alpha: feature selection threshold; p-value must be lower than this to add to model. Set to 1 to force selection.
  #' @return : lm_step. The lm_step model, having been fit stepwise. 

  # init trace
  backtrace = list()
  for (iter in 1:lm_step$p){
    step_results = evaluate_single_step(lm_step)
    # save trace
    backtrace[[iter]] = step_results
    if (step_results[2] <= alpha){
      beta_step = fit_ols_successive(lm_step, c(lm_step$X_sel, step_results[1]))
      # update selected/unselected directory
      # TODO: test to make sure this is unique/not duplicated
      lm_step$X_sel = append(lm_step$X_sel, step_results[1])
      lm_step$X_nonsel = setdiff(lm_step$X_nonsel, step_results[1])
      # update beta_step
      lm_step$beta_step[iter, 1:length(beta_step)] = beta_step
      # update number of features saved in model
      lm_step$k = lm_step$k + 1
    }
    else{
      lm_step$is_fit = T
      break
    }
  }
  lm_step$backtrace = backtrace
  lm_step
}

get_coef_matrix <- function(lm_step){
  #' Gets coefficients corresponding to a particular step
  #' @param lm_step: a fitted lm_step object
  #' @return : numeric[p, p] A matrix of coefficients (data presumed scaled, intercept omitted)
  coefs = lapply(1:lm_step$p, function(j){
    full_vec = rep(0, lm_step$p);
    for (feature in 1:min(j, lm_step$k)){
      full_vec[lm_step$X_sel[feature]] = lm_step$beta_step[min(j, lm_step$k), feature]
    };
    full_vec;
  }) %>%
    do.call("rbind", .) 
  coefs = coefs[1:lm_step$k, ] %>% t()
  coefs
}

predict_lm_step <- function(lm_step, newdata, k=NULL){
  #' Generates predictions for all steps along stepwise model
  #' @param lm_step: a fitted lm_step model
  #' @param newdata: numeric[n, p] a full matrix of prediction features
  #' @return : numeric[n, p] a matrix of each step's yhat

  # TODO: error if model not fit
  coefs = get_coef_matrix(lm_step)
  yhats = newdata %*% coefs + mean(lm_step$y)
  yhats
}

# example -----------------------------------------------------------------------

data(mtcars)
nrow(mtcars)

X = mtcars[, 2:ncol(mtcars)] %>% as.matrix() %>% unname()
X = lapply(1:ncol(X), function(k) (X[, k] - mean(X[, k]))/1) %>% do.call("cbind", .) #sd(X[, k])) %>% do.call("cbind", .)
y = mtcars %>% pull(mpg)

lm_step = lm_step_construct(X, y)
lm_step = select_stepwise(lm_step, alpha=1)

get_coef_matrix(lm_step)



