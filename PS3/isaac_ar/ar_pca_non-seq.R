library(dplyr)
library(glmnet)
library(glmnetUtils)
library(stringr)

source("ingest.R")
source("constants.R")
source("utils.R")
source("fv_util.R")

CONT_COLNAMES = setdiff(CONT_COLNAMES, "date_idx")

NSDF = 6
LAG = 31

COLS_ACTIVE = c("response", 
                # paste0("response", 1:LAG), 
                CONT_COLNAMES, 
                "county", 
                "date_idx", 
                do.call('paste0', expand.grid(CONT_COLNAMES, 1:10)))

training_data = load_training_data()

dev_data = training_data[DATES_DEV]
test_data = training_data[DATES_HOLDOUT]


grid = expand.grid(ncomp = c(50, 300),
                   nsdf = c(5, 15))

lapply(1:nrow(grid), function(j){
  cat(' - tune number: ', j, '/', nrow(grid), '\n')
  # extract the hyperparams for this iteration
  ncomp = grid$ncomp[j] # number of components
  nsdf = grid$nsdf[j]
  
  # iterate over folds
  lapply(1:length(dev_data), function(i){
    # make appropriate lags
    train_df = dev_data[[i]][[1]] %>%
      dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
      compile_lag(., LAG) %>%
      dplyr::select_at(COLS_ACTIVE);
    
    # include past for lag
    predict_df = bind_rows(dev_data[[i]][[1]], dev_data[[i]][[2]]) %>%
      dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
      compile_lag(., LAG) %>%
      dplyr::select_at(COLS_ACTIVE)%>%
      filter(date_idx == max(date_idx))
    
    decomp = prcomp(train_df %>% select_at(setdiff(COLS_ACTIVE, c("county", "date_idx", "response"))), center=T, scale.=T)
    train_df_pc = bind_cols(train_df[, c("response", "county", "date_idx")], data.frame(decomp$x))
    predict_df_pc = bind_cols(predict_df[, c("response", "county", "date_idx")], data.frame(predict(decomp, newdata=predict_df)))
    
    
    result = fit_predict(
      train_df=train_df_pc, # fit_and_predict() will scale for us
      predict_df=predict_df_pc, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
      # predict using glmnet formula
      model_func=lm,
      # predict.glmnet outputs one prediction for each lambda, so each shall be a column
      predict_func = function(fit, newdata){
        result = bind_cols(newdata, data.frame(yhat = predict(fit, newdata=newdata)));
        result
      },
      # we will scale these continuous column names
      scale_cols=paste0("PC", 1:ncomp),
      # it's fixed
      formula=as.formula(paste0(
        "response ~ county * ns(date_idx, df=nsdf) + " %>% str_replace_all(., "nsdf", as.character(nsdf)),
        paste0("PC", 1:ncomp, collapse = ' + ')
      )
      )
    ) 
    result %>%
      mutate(fold_idx = i, 
             date = names(dev_data)[i],
             ncomp = ncomp,
             nsdf=nsdf)
    
  }) %>%
    `names<-`(names(dev_data)) -> fold_result
  fold_result
}) -> result_tune


yhat_df = lapply(result_tune, function(tune) do.call("rbind", tune)) %>%
  do.call("rbind", .) %>%
  group_by(ncomp, nsdf) %>%
  do(score_loss(.)) %>%
  arrange(loss)





# NCOMP -------------------------------------------------------------------

# NCOMP = yhat_df$ncomp[1]
NCOMP = 50
NSDF = 5


lapply(1:length(test_data), function(i){
  print(i)
  # extract training and prediction data for that fold
  # ALWAYS TRAINING ON FIRST DAY
  train_df = test_data[[1]][[1]] %>%
    dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
    compile_lag(., LAG) %>%
    dplyr::select_at(COLS_ACTIVE);
  
  predict_df = bind_rows(test_data[[i]][[1]], test_data[[i]][[2]])%>%
    dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
    compile_lag(., LAG) %>%
    dplyr::select_at(COLS_ACTIVE) %>%
    filter(date_idx == max(date_idx))
  
  decomp = prcomp(train_df %>% select_at(setdiff(COLS_ACTIVE, c("county", "date_idx", "response"))), center=T, scale.=T)
  train_df_pc = bind_cols(train_df[, c("response", "county", "date_idx")], data.frame(decomp$x))
  predict_df_pc = bind_cols(predict_df[, c("response", "county", "date_idx")], data.frame(predict(decomp, newdata=predict_df)))
  
  # fit and predict for fold
  result = fit_predict(
    train_df=train_df_pc, # fit_and_predict() will scale for us
    predict_df=predict_df_pc, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
    # predict using glmnet formula
    model_func=lm, # used to fit model
    # predict.glmnet outputs one prediction for each lambda, so each shall be a column
    predict_func = function(fit, newdata){bind_cols(newdata, data.frame(yhat = predict(fit, newdata=newdata)))},
    # we will scale these continuous column names
    scale_cols=paste0("PC", 1:NCOMP),
    # the formula to be used
    formula=as.formula(paste0("response ~ county * ns(date_idx, df=", as.character(NSDF), ") + ", paste0("PC", 1:NCOMP, collapse=' + ')))
  ) 
  result %>%
    mutate(fold_idx = i) # for filtering purposes later
}) %>%
  `names<-`(names(test_data)) -> preds_pca


prediction_df = do.call("rbind", preds_pca) %>%
  dplyr::select(county, date_idx, yhat, response) 

score_loss(prediction_df)
