library(dplyr)
library(glmnet)
library(glmnetUtils)
library(splines)
source("ingest.R")
source("constants.R")
source("utils.R")
source("fv_util.R")

training_data = load_training_data()
dev_data = training_data[DATES_DEV]
test_data = training_data[DATES_HOLDOUT]

NSDF = 6

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



# Tuning ------------------------------------------------------------------

grid_lag = data.frame(c(1:10, 31)) %>% 
  t() %>%
  data.frame() %>%
  tidyr::nest(., cols = paste0("X", 1:length(.))) %>%
  `colnames<-`(c('lag'))

# forward validation
lapply(1:nrow(grid_lag), function(j){
  cat(' - tune number: ', j, '/', nrow(grid_lag), '\n')
  
  # iterate over just the last fold; perhaps we 
  lapply(length(DATES_DEV), function(i){
    
    lag_vec = grid_lag$lag[j][[1]] %>% as.numeric()
    # extract training and prediction data for that fold
    # then build lags
    train_df = dev_data[[i]][[1]] %>%
      select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
      compile_lag(., lag_vec=lag_vec) %>%
      select(-any_of(paste0("response", 1:30))); # we won't have these available
    
    # do the same for the prediction data
    predict_df = bind_rows(dev_data[[i]][[1]], dev_data[[i]][[2]]) %>%
      select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
      compile_lag(., lag_vec=lag_vec) %>%
      filter(date_idx == max(date_idx)) %>%
      select(-any_of(paste0("response", 1:30))); # we won't have these available for December
    
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
      mutate(fold_idx = i, date=DATES_DEV[j]) # for filtering purposes later
  })  -> fold_result
}) -> result_stepwise


save.image("lag-31-stepwise.RData")



# Fit (Sequential) ---------------------------------------------------------------------
STEPWISE_FORMULA = result_stepwise[[2]][[1]] %>% attr(., "formula")
STEPWISE_FORMULA = paste0("response ~ ", STEPWISE_FORMULA[3])
LAG_TUNED = grid_results$lag[1]

# extract trainX and build lags
train_df = bind_rows(dev_data[[length(dev_data)]][[1]], dev_data[[length(dev_data)]][[2]]) %>%
  compile_lag(., lb=LAG_TUNED)

mod_lm = lm(as.formula(STEPWISE_FORMULA), data=train_df)

predict_df_seq = test_data[[1]][[1]] # everything available prior to test; we'll add as we go
predictions = list()

for (i in 1:length(test_data)){
# lapply(1:length(test_data), function(i){
  # do the same for the prediction data
  # drop response here
  pred_block = test_data[[i]][[2]] %>% select(- response)
  predict_df = bind_rows(predict_df_seq, pred_block) %>%
    select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
    compile_lag(., LAG_TUNED) %>%
    filter(date_idx == min(pred_block$date_idx))
  # put in the synthetic variable
  pred_block$response = predict(mod_lm, newdata = predict_df) 
  predict_df_seq = bind_rows(predict_df_seq, pred_block) # add synthetic yhats to prediction block
  predictions[[i]] = pred_block
}

prediction_df = do.call("rbind", predictions) %>%
  select(county, date_idx, response) %>%
  left_join(bind_rows(test_data[[length(test_data)]][[1]], test_data[[length(test_data)]][[2]]) %>%
              select(county, date_idx, response),
            by=c("county", "date_idx"),
            suffix=c(".p", ""))

score_loss(prediction_df, "response.p", "response")

