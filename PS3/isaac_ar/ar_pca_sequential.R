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
                paste0("response", 1:LAG), 
                CONT_COLNAMES, 
                "county", 
                "date_idx", 
                do.call('paste0', expand.grid(CONT_COLNAMES, 1:10)))

training_data = load_training_data()

dev_data = training_data[DATES_DEV]
test_data = training_data[DATES_HOLDOUT]


grid = expand.grid(ncomp = c(150, 200, 250, 300, 350))
grid = expand.grid(ncomp = c(350))

lapply(1:nrow(grid), function(j){
  cat(' - tune number: ', j, '/', nrow(grid), '\n')
  # extract the hyperparams for this iteration
  ncomp = grid$ncomp[j] # number of components
  
  # iterate over folds
  lapply(1:length(dev_data), function(i){
    # make appropriate lags
    train_df = dev_data[[i]][[1]] %>%
      dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
      compile_lag(., LAG) %>%
      dplyr::select_at(COLS_ACTIVE);
    
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
        "response ~ county + ns(date_idx, df=NSDF) + ",
        paste0("PC", 1:ncomp, collapse = ' + ')
      )
      )
    ) 
    result %>%
      mutate(fold_idx = i, 
             date = names(dev_data)[i],
             ncomp = ncomp)
    
  }) %>%
    `names<-`(names(dev_data)) -> fold_result
  fold_result
}) -> result_tune


yhat_df = lapply(result_tune, function(tune) do.call("rbind", tune)) %>%
  do.call("rbind", .) %>%
  group_by(ncomp) %>%
  do(score_loss(.)) %>%
  arrange(loss)





# NCOMP -------------------------------------------------------------------

NCOMP = yhat_df$ncomp[1]


train_df = bind_rows(dev_data[[length(dev_data)]][[1]], dev_data[[length(dev_data)]][[2]]) %>%
  dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
  compile_lag(., LAG) %>%
  dplyr::select_at(COLS_ACTIVE);

predict_df_seq = test_data[[1]][[1]]%>%
  dplyr::select_at(c(CONT_COLNAMES, "response", "county", "date_idx")) %>%
  compile_lag(., LAG) %>%
  dplyr::select_at(COLS_ACTIVE)

decomp = prcomp(train_df %>% select_at(setdiff(COLS_ACTIVE, c("county", "date_idx", "response"))), center=T, scale.=T)
train_df_pc = bind_cols(train_df[, c("response", "county", "date_idx")], data.frame(decomp$x))

mod_lm = lm(as.formula(paste0("response ~ county + ns(date_idx, df=NSDF) + ", paste0("PC", 1:NCOMP, collapse = ' + '))), data=train_df_pc)



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
    compile_lag(., LAG) %>%
    dplyr::select_at(COLS_ACTIVE)%>%
    filter(date_idx == min(pred_block$date_idx))
  predict_df_pc = bind_cols(predict_df[, c("response", "county", "date_idx")], data.frame(predict(decomp, newdata=predict_df)))
  # put in the synthetic variable
  pred_block$response = predict(mod_lm, newdata = predict_df_pc) 
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
