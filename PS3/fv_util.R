library(dplyr)
library(glmnet)
library(glmnetUtils)
library(gam)
library(stringr)

source("ingest.R")
source("constants.R")
source("utils.R")

folds.predict <- function(data, model_func, predict_func, param, ...) {
  #' @param data Input data.frame with training and dev split into K folds
  #' @param model_func Function that generates the model
  #' @param predict_funct Function that predict the model output.
  #' @param param The hyperparameter of interest in the cross validation
  #' @param ... additional parameters to model_func
  lapply(1:length(data), function(i){
    cat("Fold ", i, "; ")
    # extract training and prediction data for that fold
    train_df = data[[i]][[1]] %>%
      select_at(c(CONT_COLNAMES, "response"));
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

forward.val <- function(hp.grid, dev_data, model_func, predict_func, ...) {
  #' @param hp.grid a grid of hyperparameters to test, called "param"
  #' @param dev_data Input data.frame with training and dev split into K folds
  #' @param model_func Function that generates the model
  #' @param predict_func Function that predict the model output.
  #' @param ... additional parameters to model_func
  lapply(1:nrow(hp.grid), function(j){
    cat(' - tune number: ', j, '/', nrow(grid_lambda), '\n')
    # extract the hyperparams for this iteration
    pj = hp.grid$param[j]
    # iterate over folds
    folds.predict(dev_data, model_func, predict_func, pj, ...)
  })
}

# Extract Results from Forward Validation
summarize.result <- function(result, hp.grid) {
  #' @param result reulst from forward validation
  #' @param hp.grid a grid of hyperparameters to test, called "param"
  grid_results = extract_folds_outer(result, hp.grid, yhat_cols = paste0("s", 48:50), collapse_func=score_loss) %>% 
    group_by(hyperparam, param) %>%
    summarise(loss = mean(loss)) %>% 
    arrange(loss)
}

# Complete Prediction Function
full.predict.score <- function(hp.grid, dev_data, test_data, 
                               model_func= function(param, data) {lm(response ~ ., data=data)}, 
                               predict_func, ...) {
  #' @param hp.grid      a grid of hyperparameters to test, called "param"
  #' @param dev_data     input data.frame with training and dev split into K folds
  #' @param test_data    data.frame against which to assess the model
  #' @param model_func   Function that generates the model. Must use a wrapper that
  #'        includes the argument "param" for the hyperparameter that will be passed
  #' @param predict_func Function that predict the model output. 
  
  result <- forward.val(hp.grid, dev_data, model_func, predict_func, ...)
  summary <- summarize.result(result, hp.grid)
  best_param <- summary$param[1]
  test_preds <- folds.predict(test_data, model_func, predict_func, param, ...)
  extract_folds_inner(test_preds, yhat_cols = summary$hyperparam[1], collapse_func=score_loss)
}

# Execution ---------------------------------------------------------

# (Step 0) Preprocess the data

#Read in the data
full_df = read.csv("./data/training_data_processed.csv")
full_df = full_df %>%
  left_join(., data.frame(date = unique(full_df$date), date_idx = 1:length(unique(full_df$date))),
            by=c("date"))

# Configure cross validation folds for the full dataset
full_df %>%
  dplyr::select(-X, -X.1) %>% # comment this out if necessary
  configure_folds() -> training_data

# Split into training/dev data and holdout test data.
dev_data = training_data[DATES_DEV]
test_data = training_data[DATES_HOLDOUT]

# (Step 1) Define the grid of hyperparameters to test
grid_df = expand.grid(
  "param" = c(2:5)
)

# (Step 2) Define a wrapper function for creating the model with argument "param"
model_func <- function(param, data) {
  formula <- response ~ . + ns(date_idx, df = param)
  lm(formula, data=data)
}

# (Step 3) Define a wrapper function for predicting response from the model
predict_func <- function(mod, newdata){
  newdata$yhat=stats::predict(mod, newdata=newdata)
  newdata
}

score <- full.predict.score(grid_df, dev_data, test_data, model_func, predict_func)