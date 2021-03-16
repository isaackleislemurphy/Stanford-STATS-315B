library(dplyr)
library(glmnet)
library(glmnetUtils)
library(stringr)

source("ingest.R")
source("constants.R")
source("utils.R")
source("fv_util.R")

CONT_COLNAMES = setdiff(CONT_COLNAMES, "date_idx")

training_data = load_training_data()

step_data = training_data[[DATES_DEV[43]]]
dev_data = training_data[DATES_DEV[44:63]]
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
    direction='forward',
    steps=3
  )
  fit_step
}



# Tuning ------------------------------------------------------------------

grid_lag = data.frame(lag = c(31))

# forward validation
lapply(1:nrow(grid_lag), function(j){
  cat(' - tune number: ', j, '/', nrow(grid_lag), '\n')
  
  cols_active = c("response", 
                  paste0("response", 1:grid_lag$lag[j]), 
                  CONT_COLNAMES, 
                  "county", 
                  "date_idx", 
                  do.call('paste0', expand.grid(CONT_COLNAMES, 1:10)))
  
  train_df = step_data[[1]] %>%
    dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
    compile_lag(., grid_lag$lag[j]) %>%
    dplyr::select_at(cols_active);
  
  predict_df = bind_rows(step_data) %>%
    dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
    compile_lag(., grid_lag$lag[j]) %>%
    dplyr::select_at(cols_active)%>%
    filter(date_idx == max(date_idx))
  
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
      result
    },
    # we will scale these continuous column names
    scale_cols=cols_active %>% setdiff(c("response", "date_idx", "county")),
    # it's fixed
    formula=NULL
  ) 
  
  result
}) -> tune_step



# PRUNE THE STEPWISE FIT
COLS_ACTIVE = c("response", 
                paste0("response", 1:LAG_TUNED), 
                CONT_COLNAMES, 
                "county", 
                "date_idx", 
                do.call('paste0', expand.grid(CONT_COLNAMES, 1:10)))
full_formula = attr(tune_step[[1]], "formula")
full_formula = readRDS("./isaac_ar/objects/step_formula.RDS")
step_order = str_split(full_formula[3], " \\+ ")[[1]]
grid2 = data.frame(K = 3:length(step_order)) # + 2 for guaranteed columns
LAG_TUNED = 31  # grid_results$lag[1]

lapply(1:nrow(grid2), function(j){
  cat(' - tune number: ', j, '/', nrow(grid2), '\n')
  # extract the hyperparams for this iteration
  K = grid2$K[j] # num features
  
  # iterate over folds
  lapply(1:length(dev_data), function(i){
    # make appropriate lags
    train_df = step_data[[1]] %>%
      dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
      compile_lag(., LAG_TUNED) %>%
      dplyr::select_at(COLS_ACTIVE);
    
    predict_df = bind_rows(step_data) %>%
      dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
      compile_lag(., LAG_TUNED) %>%
      dplyr::select_at(COLS_ACTIVE)%>%
      filter(date_idx == max(date_idx))
    
    result = fit_predict(
      train_df=train_df, # fit_and_predict() will scale for us
      predict_df=predict_df, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
      # predict using glmnet formula
      model_func=lm,
      # predict.glmnet outputs one prediction for each lambda, so each shall be a column
      predict_func = function(fit, newdata){
        result = bind_cols(newdata, data.frame(yhat = predict(fit, newdata=newdata)));
        result
      },
      # we will scale these continuous column names
      scale_cols=cols_active %>% setdiff(c("response", "date_idx", "county")),
      # it's fixed
      formula=as.formula(paste0(
        "response ~ ",
        paste0(step_order[1:K], collapse = ' + ')
        )
      )
    ) 
    result %>%
      mutate(fold_idx = i, 
             date = names(dev_data)[i], 
             model_formula = paste0("response ~ ", paste0(step_order[1:K], collapse = ' + ')))
    
  }) %>%
    `names<-`(names(dev_data)) -> fold_result
  fold_result
}) -> result_prune


yhat_df = lapply(result_prune, function(tune) do.call("rbind", tune)) %>%
  do.call("rbind", .) %>%
  group_by(model_formula) %>%
  do(score_loss(.)) %>%
  arrange(loss)

PRUNED_FORMULA = yhat_df %>% pull(model_formula) %>% head(1)
PRUNED_FORMULA = paste0("response ~ ", paste0(step_order, collapse = ' + '))




# Fit (Sequential) ---------------------------------------------------------------------

# extract trainX and build lags
# use the last of the devdata available

train_df = bind_rows(dev_data[[length(dev_data)]][[1]], dev_data[[length(dev_data)]][[2]]) %>%
  dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
  compile_lag(., LAG_TUNED) %>%
  dplyr::select_at(COLS_ACTIVE);

mod_lm = lm(as.formula(PRUNED_FORMULA), data=train_df)

predict_df_seq = test_data[[1]][[1]]%>%
  dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
  compile_lag(., LAG_TUNED) %>%
  dplyr::select_at(COLS_ACTIVE)

predictions = list()

for (i in 1:length(test_data)){
# lapply(1:length(test_data), function(i){
  # do the same for the prediction data
  # drop response here
  
  # get the next batch of predictions on deck
  pred_block = test_data[[i]][[2]] %>% dplyr::select(- response)
  # make lags of them
  predict_df = bind_rows(predict_df_seq, pred_block) %>%
    dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
    compile_lag(., LAG_TUNED) %>%
    dplyr::select_at(COLS_ACTIVE)%>%
    filter(date_idx == min(pred_block$date_idx))
  # put in the synthetic variable
  pred_block$response = predict(mod_lm, newdata = predict_df) 
  predict_df_seq = bind_rows(predict_df_seq, pred_block) # add synthetic yhats to prediction block
  predictions[[i]] = pred_block
}

prediction_df = do.call("rbind", predictions) %>%
  dplyr::select(county, date_idx, response) %>%
  left_join(bind_rows(test_data[[length(test_data)]][[1]], test_data[[length(test_data)]][[2]]) %>%
              dplyr::select(county, date_idx, response),
            by=c("county", "date_idx"),
            suffix=c(".p", ""))

score_loss(prediction_df, "response.p", "response")

