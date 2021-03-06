library(dplyr)
library(glmnet)
library(glmnetUtils)
library(splines)

source("ingest.R")
source("constants.R")
source("utils.R")

CONT_COLNAMES = setdiff(CONT_COLNAMES, "date_idx")

training_data = load_training_data()

dev_data = training_data[DATES_DEV]
test_data = training_data[DATES_HOLDOUT]


# Baseline 1: Tuning L1/L2 -------------------------------------------------

# tuning grid
grid_lambda = expand.grid(
  "alpha" = seq(0, .8, .4)
)

# forward validation
lapply(1:nrow(grid_lambda), function(j){
  cat(' - tune number: ', j, '/', nrow(grid_lambda), '\n')
  # extract the hyperparams for this iteration
  a = grid_lambda$alpha[j]
  # iterate over folds
  lapply(1:length(dev_data), function(i){
    # extract training and prediction data for that fold
    train_df = dev_data[[i]][[1]] %>%
      select_at(c(CONT_COLNAMES, "response"));
    predict_df = dev_data[[i]][[2]];
    # fit and predict for fold
    result = fit_predict(
      train_df=train_df, # fit_and_predict() will scale for us
      predict_df=predict_df, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
      # predict using glmnet formula
      model_func=glmnetUtils:::glmnet.formula, # used to fit model
      # predict.glmnet outputs one prediction for each lambda, so each shall be a column
      predict_func = function(fit, newdata){bind_cols(newdata, data.frame(glmnetUtils:::predict.glmnet.formula(fit, newdata=newdata)))},
      # we will scale these continuous column names
      scale_cols=CONT_COLNAMES,
      # the formula to be used
      formula=as.formula("response~."),
      # mixing hyperparam; glmnet chooses a reasonable grid of alphas
      alpha=a
    ) 
    result %>%
      mutate(fold_idx = i) # for filtering purposes later
  }) %>%
    `names<-`(names(dev_data)) -> fold_result
}) -> result_glmnet_demo

# extract FV results
grid_results = extract_folds_outer(result_glmnet_demo, grid_lambda, yhat_cols = paste0("s", 48:50), collapse_func=score_loss) %>% 
  group_by(hyperparam, alpha) %>%
  summarise(loss = mean(loss)) %>% 
  arrange(loss)

# use the best tune to make predictions
lapply(1:length(test_data), function(i){
  # extract training and prediction data for that fold
  train_df = test_data[[i]][[1]] %>%
    select_at(c(CONT_COLNAMES, "response"));
  predict_df = test_data[[i]][[2]];
  # fit and predict for fold
  result = fit_predict(
    train_df=train_df, # fit_and_predict() will scale for us
    predict_df=predict_df, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
    # predict using glmnet formula
    model_func=glmnetUtils:::glmnet.formula, # used to fit model
    # predict.glmnet outputs one prediction for each lambda, so each shall be a column
    predict_func = function(fit, newdata){bind_cols(newdata, data.frame(glmnetUtils:::predict.glmnet.formula(fit, newdata=newdata)))},
    # we will scale these continuous column names
    scale_cols=CONT_COLNAMES,
    # the formula to be used
    formula=as.formula("response~."),
    # mixing hyperparam; glmnet chooses a reasonable grid of alphas
    alpha=grid_results$alpha[1]
  ) 
  result %>%
    mutate(fold_idx = i) # for filtering purposes later
}) %>%
  `names<-`(names(test_data)) -> test_preds # rbind test preds if you wish to compare

grid_results$alpha
# prediction_scoring
ols_score = extract_folds_inner(test_preds, yhat_cols = grid_results$hyperparam[1], collapse_func=score_loss)
sum(ols_score["loss"])
# Baseline loss is 14.15

# Baseline 2: PCA ---------------------------------------------------------

# tuning grid
grid_ncomp = expand.grid(
  # "K" = c(seq(2:length(CONT_COLNAMES))),
  "K" = seq(20, 30, 5),
  "df" = c(5, 10, 15)
)

# forward validation
lapply(1:nrow(grid_ncomp), function(j){
  cat(' - tune number: ', j, '/', nrow(grid_ncomp), '\n')
  # extract the hyperparams for this iteration
  K = grid_ncomp$K[j]
  DF = grid_ncomp$df[j]
  # iterate over folds
  lapply(1:length(dev_data), function(i){
    # extract training and prediction data for that fold
    decomp = prcomp(dev_data[[i]][[1]][, CONT_COLNAMES], rank = K, scale. = T, center = T);
    train_df = cbind(data.frame(decomp$x), dev_data[[i]][[1]][, c("county", "response", "date_idx")]);
    
    predict_df = cbind(
      predict(object=decomp, newdata = dev_data[[i]][[2]]),
      dev_data[[i]][[2]][, c("county", "response", "date_idx", "date")]
    )
    
    # fit and predict for fold
    result = fit_predict(
      train_df=train_df, # fit_and_predict() will scale for us
      predict_df=predict_df, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
      # predict using glmnet formula
      model_func=lm, # used to fit model
      # predict.glmnet outputs one prediction for each lambda, so each shall be a column
      predict_func = function(fit, newdata){bind_cols(newdata, data.frame(yhat = predict(fit, newdata=newdata)))},
      # we will scale these continuous column names
      scale_cols=setdiff(colnames(train_df), c("response", "county", "date_idx")),
      # the formula to be used
      formula=as.formula(paste0("response ~ county*ns(date_idx, df=", as.character(DF), ") + ", paste0("PC", 1:K, collapse=' + ')))
    ) 
    result %>%
      mutate(fold_idx = i, ncomp=K, df = DF) # for filtering purposes later
  }) %>%
    `names<-`(names(dev_data)) -> fold_result
  fold_result
}) -> result_pca_demo

# extract FV results
grid_results = extract_folds_outer(result_pca_demo, grid_ncomp, yhat_cols = "yhat", collapse_func=score_loss) %>% 
  group_by(hyperparam, K) %>%
  summarise(loss = mean(loss)) %>% 
  arrange(loss)

NCOMP = grid_results$K[1]
NSDF = grid_results$df[1]

NCOMP = 25
NSDF = 6

# use the best tune to make predictions
lapply(1:length(test_data), function(i){
  # extract training and prediction data for that fold
  decomp = prcomp(test_data[[1]][[1]][, CONT_COLNAMES], rank = NCOMP, scale. = T, center = T);
  train_df = cbind(data.frame(decomp$x), test_data[[1]][[1]][, c("county", "response", "date_idx")]);
  
  predict_df = cbind(
    predict(object=decomp, newdata = test_data[[i]][[2]]),
    test_data[[i]][[2]][, c("county", "response", "date", "date_idx")]
  )
  
  # fit and predict for fold
  result = fit_predict(
    train_df=train_df, # fit_and_predict() will scale for us
    predict_df=predict_df, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
    # predict using glmnet formula
    model_func=lm, # used to fit model
    # predict.glmnet outputs one prediction for each lambda, so each shall be a column
    predict_func = function(fit, newdata){bind_cols(newdata, data.frame(yhat = predict(fit, newdata=newdata)))},
    # we will scale these continuous column names
    scale_cols=setdiff(colnames(train_df), c("response", "county", "date_idx")),
    # the formula to be used
    formula=as.formula(paste0("response ~ county * ns(date_idx, df=", as.character(NSDF), ") + ", paste0("PC", 1:NCOMP, collapse=' + ')))
  ) 
  result %>%
    mutate(fold_idx = i) # for filtering purposes later
}) %>%
  `names<-`(names(test_data)) -> preds_pca # rbind test preds if you wish to compare

# prediction_scoring
pca_score = extract_folds_inner(preds_pca, yhat_cols = "yhat", collapse_func=score_loss)
mean(pca_score$loss)
# Best K for PCA = 5 / 31
# Total loss for November Prediction = 7.934045