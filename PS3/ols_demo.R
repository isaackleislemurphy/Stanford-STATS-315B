library(dplyr)
library(glmnet)
library(glmnetUtils)
source("ingest.R")
source("constants.R")
source("utils.R")

training_data = read.csv("./data/training_data_processed.csv") %>%
  select(-X, -X.1) %>% # comment this out if necessary
  configure_folds()


# Example 1: Straightaway Fit ---------------------------------------------

# then wrap this in a loop/grid of hyperparameters
lapply(1:length(training_data), function(i){
  train_df = training_data[[i]][[1]] %>%
    select_at(c(CONT_COLNAMES, "response"));
  predict_df = training_data[[i]][[2]];
  result = fit_predict(
    train_df=train_df,
    predict_df=predict_df,
    scale_cols=CONT_COLNAMES,
    formula = as.formula("response~.")
  );
  result
}) -> lm_example


# Example 2: Tuning L1/L2 -------------------------------------------------

grid_lambda = expand.grid(
  "alpha" = c(1e-1, 1e-2, 1e-3, 1e-4)
)

lapply(1:nrow(grid_lambda), function(j){
  cat(' - tune number: ', j, '/', nrow(grid_lambda), '\n')
  # extract the hyperparams for this iteration
  a = grid_lambda$alpha[j]
  # iterate over folds
  lapply(1:length(training_data), function(i){
    # extract training and prediction data for that fold
    train_df = training_data[[i]][[1]] %>%
      select_at(c(CONT_COLNAMES, "response"));
    predict_df = training_data[[i]][[2]];
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
  })  -> fold_result
})


