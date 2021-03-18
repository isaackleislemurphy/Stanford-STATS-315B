library(dplyr)
library(gam)
library(glmnet)
library(glmnetUtils)
library(mgcv)
library(splines)


source("constants.R")
CONT_COLNAMES=setdiff(CONT_COLNAMES,"date_idx")
source("fv_util.R")
source("ingest.R")
source("utils.R")

training_data = load_training_data()
dev_data = training_data[DATES_DEV]
test_data = training_data[DATES_HOLDOUT]

grid_df = expand.grid(
  "df" = seq(2, 20, 1)
)

# forward validation
lapply(1:nrow(grid_df), function(j){
  cat(' - tune number: ', j, '/', nrow(grid_df), '\n')
  # extract the hyperparams for this iteration
  deg_free = grid_df$df[j]
  # iterate over folds
  lapply(1:length(dev_data), function(i){
    # extract training and prediction data for that fold
    train_df = dev_data[[i]][[1]] %>%
      dplyr::select_at(c(CONT_COLNAMES, "county", "response","date_idx"));
    predict_df = dev_data[[i]][[2]];
    # fit and predict for fold
    result = fit_predict(
      train_df=train_df, # fit_and_predict() will scale for us
      predict_df=predict_df, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
      # predict using glmnet formula
      model_func=lm,
      # predict.glmnet outputs one prediction for each lambda, so each shall be a column
      predict_func = function(fit, newdata){bind_cols(newdata, data.frame(yhat=predict(fit, newdata=newdata)))},
      # we will scale these continuous column names
      scale_cols=CONT_COLNAMES,
      # the formula to be used
      formula=as.formula(paste0("response ~ county:ns(date_idx, df=", 
                                as.character(deg_free), ")"))
    )

    result %>%
      mutate(fold_idx = i) # for filtering purposes later
  }) %>%
    `names<-`(names(dev_data)) -> fold_result
}) -> result_county_spline

# extract FV results
grid_results = extract_folds_outer(result_county_spline, grid_df, yhat_cols = "yhat", collapse_func=score_loss) %>% 
  group_by(df) %>%
  summarise(loss = mean(loss)) %>% 
  arrange(loss)



# Testing on November data

# df=20 was selected

lapply(1:length(test_data), function(i){
  # extract training and prediction data for that fold
  train_df = test_data[[i]][[1]] %>%
    dplyr::select_at(c(CONT_COLNAMES, "county", "response","date_idx"));
  predict_df = test_data[[i]][[2]];
  # fit and predict for fold
  result = fit_predict(
    train_df=train_df, # fit_and_predict() will scale for us
    predict_df=predict_df, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
    # predict using glmnet formula
    model_func=lm,
    # predict.glmnet outputs one prediction for each lambda, so each shall be a column
    predict_func = function(fit, newdata){bind_cols(newdata, data.frame(yhat=predict(fit, newdata=newdata)))},
    # we will scale these continuous column names
    scale_cols=CONT_COLNAMES,
    # the formula to be used
    formula=as.formula(response ~ . +ns(date_idx, df=20))
  )
  
  result %>%
    mutate(fold_idx = i) # for filtering purposes later
}) %>%
  `names<-`(names(test_data)) -> fold_result -> result_county_spline_test





