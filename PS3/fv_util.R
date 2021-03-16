library(dplyr)
library(stringr)

source("ingest.R")
source("constants.R")
source("utils.R")

folds.predict <- function(data, model_func, predict_func, param, addl_cols=c(), ...) {
  #' @param data Input data.frame with training and dev split into K folds
  #' @param model_func Function that generates the model
  #' @param predict_funct Function that predict the model output.
  #' @param param The hyperparameter of interest in the cross validation
  #' @param ... additional parameters to model_func
  lapply(1:length(data), function(i){
    cat("Fold ", i, "; ")
    # extract training and prediction data for that fold
    train_idx = 1
    if (i > 31) {
      train_idx = i - 31
    }
    train_df = data[[train_idx]][[1]] %>%
      select_at(c(CONT_COLNAMES, addl_cols, "county", "response", "date_idx"));
    predict_df = data[[i]][[2]];
    
    # fit and predict for fold
    result = fit_predict(
      train_df=train_df, # fit_and_predict() will scale for us
      predict_df=predict_df, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
      # predict using glmnet formula
      model_func = model_func, # used to fit model
      # predict.glmnet outputs one prediction for each lambda, so each shall be a column
      predict_func = predict_func,
      # we will scale these continuous column names
      scale_cols = CONT_COLNAMES,
      # additional params for the model, such as formula
      param = param,
      ...
    ) 
    cat("\n")
    result %>%
      mutate(fold_idx = i) # for filtering purposes later
  }) %>%
    `names<-`(names(data)) -> result
}

forward.val <- function(hp.grid, dev_data, model_func, predict_func, addl_cols=c(), ...) {
  #' @param hp.grid a grid of hyperparameters to test, called "param"
  #' @param dev_data Input data.frame with training and dev split into K folds
  #' @param model_func Function that generates the model
  #' @param predict_func Function that predict the model output.
  #' @param ... additional parameters to model_func
  lapply(1:nrow(hp.grid), function(j){
    cat(' - tune number: ', j, '/', nrow(hp.grid), '\n')
    # extract the hyperparams for this iteration
    pj = hp.grid$param[j]
    # iterate over folds
    folds.predict(dev_data, model_func, predict_func, pj, addl_cols, ...)
  })
}

# Extract Results from Forward Validation
summarize.result <- function(result, hp.grid, yhat_cols) {
  #' @param result result from forward validation
  #' @param hp.grid a grid of hyperparameters to test, called "param"
  grid_results = extract_folds_outer(result, hp.grid, yhat_cols = yhat_cols, collapse_func=score_loss) %>% 
    group_by(hyperparam, param) %>%
    summarise(loss = mean(loss)) %>% 
    arrange(loss)
}

# Complete Prediction Function
full.predict.score <- function(hp.grid, dev_data, test_data, 
                               model_func= function(param, data) {lm(response ~ ., data=data)}, 
                               predict_func, 
                               yhat_cols, addl_cols=c(), ...) {
  #' @param hp.grid      a grid of hyperparameters to test, called "param"
  #' @param dev_data     input data.frame with training and dev split into K folds
  #' @param test_data    data.frame against which to assess the model
  #' @param model_func   Function that generates the model. Must use a wrapper that
  #'        includes the argument "param" for the hyperparameter that will be passed
  #' @param predict_func Function that predict the model output.
  #' @param yhat_cols    Characters naming the columns output by predict_func
  #' @param ...          Additional parameters to pass to model.func 
  
  result <- forward.val(hp.grid, dev_data, model_func, predict_func, addl_cols, ...)
  summary <- summarize.result(result, hp.grid, yhat_cols)
  best_param <- summary$param[1]
  print(summary$param[1])
  test_preds <- folds.predict(test_data, model_func, predict_func, best_param, addl_cols, ...)
  list(best_param, extract_folds_inner(test_preds, yhat_cols = summary$hyperparam[1], collapse_func=score_loss))
}

# Execution ---------------------------------------------------------

# # (Step 0) Preprocess the data
# 
# #Read in the data
# full_df = read.csv("./data/training_data_processed.csv")
# full_df = full_df %>%
#   left_join(., data.frame(date = unique(full_df$date), date_idx = 1:length(unique(full_df$date))),
#             by=c("date"))
# 
# # Configure cross validation folds for the full dataset
# full_df %>%
#   dplyr::select(-X, -X.1) %>% # comment this out if necessary
#   configure_folds() -> training_data
# 
# # Split into training/dev data and holdout test data.
# dev_data = training_data[DATES_DEV]
# test_data = training_data[DATES_HOLDOUT]
# 
# # (Step 1) Define the grid of hyperparameters to test
# grid_df = expand.grid(
#   "param" = c(2:5)
# )
# 
# # (Step 2) Define a wrapper function for creating the model with argument "param"
# model_func <- function(param, data) {
#   formula <- response ~ . + ns(date_idx, df = param)
#   lm(formula, data=data)
# }
# 
# # (Step 3) Define a wrapper function for predicting response from the model
# predict_func <- function(mod, newdata){
#   newdata$yhat=stats::predict(mod, newdata=newdata)
#   newdata
# }
# 
# score <- full.predict.score(grid_df, dev_data, test_data, model_func, predict_func)