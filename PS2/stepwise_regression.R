suppressMessages(library(dplyr))

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
    x_sel_idx=c(),
    X_sel=matrix(rep(NA, nrow(X)*ncol(X)), ncol=ncol(X)), # actual columns selected
    X_nonsel_idx=1:ncol(X), # indices of columns in X not selected
    y=y, # targets
    # beta_step=matrix(rep(0, ncol(X)^2), ncol=ncol(X)), # log of betas
    # is_fit=F, # for checks
    j=1 # current predictor number
  )
  lm_init$Q = matrix(rep(0, (lm_init$p) * lm_init$n), ncol=lm_init$p)
  lm_init$R = matrix(rep(0, (lm_init$p)^2), ncol=lm_init$p)
  attr(lm_init, "class") = "lm_step"
  lm_init
}

advance_qr <- function(lm_step, new_col_idx){
  #' Performs G-S decomposition for one new column of X
  #' @param lm_step: lm_step. An lm_step object, partially fitted
  #' @param new_col_idx: integer. The index of the new column to add to the decomposition.
  #' @return : lm_step. 
  
  # GRAM SCHMIDT STEP
  ai = lm_step$X[, new_col_idx]
  if (lm_step$j == 1 & is.null(lm_step$X_sel)){
    vi = ai
  }else{
    vi = ai + lapply(1:(lm_step$j - 1), function(ll) 
      - as.numeric(t(ai) %*% lm_step$Q[, ll]) * lm_step$Q[, ll]
    ) %>% 
      do.call("cbind", .) %>%
      rowSums()
  }
  
  ei = vi/as.numeric(sqrt(t(vi) %*% vi))
  lm_step$Q[, lm_step$j] = ei
  lm_step$x_sel_idx  = append(lm_step$x_sel_idx, new_col_idx)
  lm_step$X_nonsel_idx = setdiff(lm_step$X_nonsel_idx, new_col_idx)
  lm_step$X_sel[, lm_step$j] = lm_step$X[, new_col_idx]
  lm_step$R[1:(lm_step$j), lm_step$j] = sapply(1:lm_step$j, function(ll) t(ai) %*% lm_step$Q[, ll])
  lm_step$j = lm_step$j + 1
  lm_step
}

get_beta <- function(lm_step, k=NULL){
  #' Gets the coefficients (not the intercept) from an lm_step object, retroactive to a step number
  #' @param lm_step: lm_step. A (at least partially) fitted lm_step object
  #' @param k: integer. The stepnumber to get beta retroactive to. If null, uses entirety of fit
  #' @return : numeric[k]. A vector of OLS coefficients
  y = lm_step$y
  k = ifelse(is.null(k), lm_step$j-1, k)
  Q = lm_step$Q[, 1:k]; R = lm_step$R[1:k, 1:k]
  backsolve(R, crossprod(Q, y)) %>%
    as.numeric()
}

advance_selection <- function(lm_step){
  #' Identifies which "next feature" to add, and adds that feature to a copied version of the input
  #' @param lm_step: An partially fit lm_step object
  #' @return : lm_step. The lm_step, having been advanced to include the next available feature that most reduces RSS
  #' 
  beta_current = get_beta(lm_step);
  if (length(beta_current) == 1){
    rss_current = sum((lm_step$y - mean(lm_step$y) - lm_step$X_sel[, 1:(lm_step$j-1)] * beta_current)^2)
  }else{
    rss_current = sum((lm_step$y - mean(lm_step$y) - lm_step$X_sel[, 1:(lm_step$j-1)] %*% beta_current)^2)
  }
  candidates = lapply(lm_step$X_nonsel_idx, function(ll){
    lm_step_cand = advance_qr(lm_step, ll);
    beta_cand = get_beta(lm_step_cand);
    rss_delta = rss_current - sum((lm_step_cand$y - mean(lm_step_cand$y) - lm_step_cand$X_sel[, 1:(lm_step_cand$j-1)] %*% beta_cand)^2);
    list(lm_step_cand, rss_delta);
  })
  candidates[[which.max(sapply(candidates, function(x) x[[2]]))]][[1]]
}

stepwise.fit <- function(X, y, k=NULL){
  #' Fits OLS stepwise.
  #' @param X: numeric[n, p]. A matrix of features
  #' @param y: numeric[n]. A vector of OLS targets
  #' @param k: integer. The optional number of steps to take. If NULL, takes p steps
  #' @return : lm_step. A fitted lm_step object.
  k = ifelse(is.null(k), ncol(X), k)
  lm_step = lm_step_construct(unname(as.matrix(X)), y)
  init_beta = which.min(sapply(1:ncol(lm_step$X), function(ll) 
    sum((lm_step$y - 
           mean(lm_step$y) - 
           as.numeric((t(lm_step$X[, ll]) %*% lm_step$y))/as.numeric(t(lm_step$X[, ll]) %*% lm_step$X[, ll]) * lm_step$X[, ll])^2)
  ))
  lm_step = advance_qr(lm_step, init_beta)
  for (iter in 2:k){
    lm_step = advance_selection(lm_step)
  }
  lm_step
}

stepwise.predict <- function(X, lm_step, k=NULL){
  #' Predicts from a fitted lm_step. 
  #' @param X: numeric[n1, p]. A matrix of features to predict. Note columns must be ordered identical to original input.
  #' @param lm_step: lm_step. A fitted lm_step object
  #' @param k: integer. Number of features to retroactivelly predict. If null, uses entirety of fit.
  #' @return : numeric[n1]. A vector of predictions.
  k = ifelse(is.null(k), lm_step$j-1, k)
  beta_hat = get_beta(lm_step, k=k)
  if(length(beta_hat) - 1){
    return(as.numeric(mean(lm_step$y) + unname(as.matrix(X))[, lm_step$x_sel_idx[1:k]] %*% beta_hat))
  }else{
    return(as.numeric(mean(lm_step$y) + unname(as.matrix(X))[, lm_step$x_sel_idx[1:k]] * beta_hat))
  }
}

stepwise.cv.fit(X, y, nfolds=10, k=NULL, S=2020){
  set.seed(S)
  fold_idx = sample(1:nrow(X), nrow(X), replace=F)
  folds = split(fold_idx, ceiling(seq_along(fold_idx) / ceiling(length(fold_idx)/nfolds)))
  cv_result = lapply(folds, function(fold){
    cat('-')
    X_train = X[setdiff(fold_idx, fold), ]; y_train = y[setdiff(fold_idx, fold)];
    X_dev = X[fold, ]; y_dev = y[fold];
    # center data
    centers = X_train %>% colMeans()
    # apply centers to training
    X_train = lapply(1:nrow(X_train), function(i)
      as.numeric(X_train[i, ]) - centers
    ) %>%
      do.call("rbind", .)
    X_dev = lapply(1:nrow(X_dev), function(i)
      as.numeric(X_dev[i, ]) - centers
    ) %>%
      do.call("rbind", .)
    
    model = stepwise.fit(X_train, y_train, k=k)
    scoring = lapply(1:length(model$x_sel_idx), function(z){
      yhat = stepwise.predict(X_dev, model, k=z);
      yhat_binary = sapply(yhat, function(y) ifelse(y >= .5, 1, 0));
      rss = sum((y_dev - yhat)^2);
      mse = mean((y_dev - yhat)^2);
      misclass = mean(y_dev != yhat_binary);
      c(rss, mse, misclass)
    }) %>%
      do.call("rbind", .) %>%
      data.frame() %>%
      `colnames<-`(c("RSS", "MSE", "MISCLASS"))
    scoring$nfeat = 1:length(model$x_sel_idx)
    scoring
  })
  
  cv_agg = do.call("rbind", cv_result) %>%
    group_by(nfeat) %>%
    summarise_all(., mean)
  
}


