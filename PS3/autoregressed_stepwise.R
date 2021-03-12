library(dplyr)
library(glmnet)
library(glmnetUtils)
source("ingest.R")
source("constants.R")
source("utils.R")
source("fv_util.R")

training_data = load_training_data()
dev_data = training_data[DATES_DEV]
test_data = training_data[DATES_HOLDOUT]

NSDF = 20

compile_lag <- function(df, lb=3){
  #' Builds lags
  #' @param df: data.frame. A dataframe of training data
  #' @param lb: integer. Time window to look back over, and join lags t-1, ..., t-lb. 
  #' @return : data.frame. A dataframe with time lags. 
  lag_df = df
  for (lag in 1:lb){
    lag_df = lag_df %>%
      inner_join(., df %>% mutate(date_idx = date_idx + lag),
                 by = c("county", "date_idx"),
                 suffix = c("", lag))
  }
  lag_df
}

step.mod <- function(data, ...){
  naive_model = lm(response ~ county + ns(date_idx, df=NSDF), data=data);
  fit_step = stats::step(
    object=naive_model,
    scope=as.formula(
      paste0("response ~ . +",
             paste0(data %>% colnames() %>% setdiff(., c("response", "date_idx")), collapse = ' + ')
             )
    ),
    direction='forward'
  )
  fit_step
}

grid_lag = data.frame(lag = c(5, 10))

# forward validation
lapply(1:nrow(grid_lag), function(j){
  cat(' - tune number: ', j, '/', nrow(grid_lag), '\n')
  
  # iterate over just the last fold; perhaps we 
  lapply(length(DATES_DEV), function(i){
    
    # extract training and prediction data for that fold
    # then build lags
    train_df = dev_data[[i]][[1]] %>%
      select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
      compile_lag(., grid_lag$lag[j]);
    
    # do the same for the prediction data
    predict_df = bind_rows(dev_data[[i]][[1]], dev_data[[i]][[2]]) %>%
      select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
      compile_lag(., grid_lag$lag[j]) %>%
      filter(date_idx == max(date_idx));
    
    # # fit and predict for fold
    # fit_step = step.mod(train_df);
    # step_formula = as.character(formula(fit_step$call));
    
    result = fit_predict(
      train_df=train_df, # fit_and_predict() will scale for us
      predict_df=predict_df, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
      # predict using glmnet formula
      model_func=step.mod,
      # predict.glmnet outputs one prediction for each lambda, so each shall be a column
      predict_func = function(fit, newdata){
        result = bind_cols(newdata, data.frame(yhat = predict(fit, newdata=newdata)));
        attr(result, "formula") = as.character(formula(fit$call))
        },
      # we will scale these continuous column names
      scale_cols=CONT_COLNAMES,
      # it's fixed
      formula=NULL
    ) 
    result %>%
      mutate(fold_idx = i) # for filtering purposes later
  })  -> fold_result
}) -> result_stepwise