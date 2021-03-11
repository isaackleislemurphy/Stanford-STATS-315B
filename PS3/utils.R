library(dplyr)


# Standard Scaler ---------------------------------------------------------

StandardScaler <- function(){
  #' Constructor for object that performs (x - u)/sd scaling
  #' Similar to sklearn.preprocessing.StandardScaler
  list(scale=NULL, is_fit=F, cols=NULL)
}

scaler_fit <- function(scaler, df, cols=CONT_COLNAMES){
  #' Fits scaler parameters to data and saves those parameters for later use
  #' @param scaler: StandardScaler. A StandardScaler object
  #' @param df: data.frame. A dataframe containing data for the scaler fit
  #' @param cols: character[p]. A vector of column names in `df` to perform the scaling on.
  #' @return : StandardScaler. A fitted StandardScaler object
  scaler$cols = cols
  scaler$scale = scale(df[, cols], center=T, scale=T)
  scaler$is_fit=T
  scaler
}

scaler_transform <- function(scaler, df){
  #' Transforms data according to a StandardScaler having been fit
  #' @param scaler: StandardScaler. A StandardScaler object
  #' @param df: data.frame. A dataframe containing data to be transformed. Must contain columns `cols`.
  #' @return : data.frame. A transformed/scaled dataframe
  df[, scaler$cols] = data.frame(
    scale(df[, scaler$cols],
          center=attr(scaler$scale, 'scaled:center'),
          scale=attr(scaler$scale, 'scaled:scale'))
  ) %>%
    `colnames<-`(scaler$cols)
  df
}

# Train/Predict within Fold -----------------------------------------------

fit_predict <- function(train_df, 
                        predict_df, 
                        model_func=lm, 
                        predict_func=function(mod, newdata){newdata$yhat=stats::predict(mod, newdata=newdata); newdata},
                        scale_cols=CONT_COLNAMES, 
                        ...){
  #' A function that trains on a training df and subsequently predicts on a new dataframe.
  #' Intended use is for ("fold", "dev") pairing.
  #' @param train_df: data.frame. A dataframe of training data, all of `scale_cols` unscaled at this point.
  #' @param predict_df: data.frame. A dataframe of prediction/dev data, all of `scale_cols` unscaled at this point.
  #' @param model_func: function. A function, e.g. `stats::lm()`, used to fit the model.
  #' @param predict_func: function. A wrapper around a prediction function, e.g. `stats::predict()`, used to predict from a fitted model
  #' Some functions, such as predict.glmnet, may return a matrix of yhats; hence the wrapper
  #' @param scale_cols: character[p]. A vector of continuous-valued columns to scale prior to fitting. 
  #' @param ...: args passed to predict_func. 
  #' @return : data.frame. The dataframe with fitted predictions
  scaler = StandardScaler()
  scaler = scaler_fit(scaler, df=train_df, cols=scale_cols)
  train_df_sc = scaler_transform(scaler, train_df)
  predict_df_sc = scaler_transform(scaler, predict_df)
  fit = model_func(data=train_df_sc, ...)
  predict_df = predict_func(fit, newdata=predict_df_sc)
  predict_df
}



# Scoring Helpers ---------------------------------------------------------

extract_folds_inner <- function(result, 
                                yhat_cols, 
                                ytrue_col="response", 
                                collapse_func=function(x, ...){x},
                                ...){
  #' Used to score a list of folds corresponding to one hyperparam setting, e.g. the inner loop (j) of 
  #' ols_demo. Note that in some cases, a single hyperparameter setting may come with multiple additional hyperparams
  #' for another hyperparam (e.g. glmnet, which for a single alpha automatically tries 100 different lambdas. see ?glmnet).
  #'  As such,this single hyperparam may actually contain multiple such hyperparams; this function extracts and "stacks" those results
  #'  for direct comparison
  #'  @param result : list[data.frame]. A list of dev-set projections, corresponding to folds/break points. Each item is a dataframe,
  #'  containing a "response" column with the true value and other (custom) columns with predicted columns
  #'  @param metric : character. One of 'rmse', 'mae', 'pearson', 'spearman', for scoring metric. 
  #'  @param yhat_cols : character[s]. A vector of column names corresponding to predictions for the dev set. May be multiple columns,
  #'  like in the glmnet auto-100 lambda situation. 
  #'  @param ytrue_col : character. The column name of the true observation.
  #'  @param collapse_func : function. A scoring function to collapse/score the predictions, to manage size. Must include
  #'  its own groupby clause
  lapply(1:length(result), function(j){
    lapply(yhat_cols, function(col){
      result[[j]][, c("date", ytrue_col, col)] %>%
        rename(yhat = eval(col)) %>%
        mutate(hyperparam = col,
               fold_idx=  j) # for groupby --> min purposes
    }) %>%
      do.call("rbind", .)
  }) %>%
    do.call("rbind", .) %>% 
    group_by(hyperparam, fold_idx, date) %>%
    do(collapse_func(., ...)) -> result_stacked
  result_stacked
}

extract_folds_outer <- function(result, tunegrid, ...){
  #' Used to extract/stack results over the "outer" grid, i.e. the one provided outside
  #' of the fit function. See ols_demo.R --> grid_lambda
  lapply(1:nrow(tunegrid), function(i){
    extract_folds_inner(result[[i]], ...) -> result_outer;
    # add the hyperparams corresponding to grid row
    for (gridcol in colnames(tunegrid)){
      result_outer[gridcol] = tunegrid[i, gridcol]
    }
    result_outer
  }) %>%
    do.call("rbind", .)
}

score_loss <- function(df, yhat_col='yhat', ytrue_col='response'){
  df = df %>% mutate_at(yhat_col, function(x) ifelse(x <= 0, 1e-20, x))
  loss = abs(
    log(1 + df[, ytrue_col]) - log(1 + df[, yhat_col])
  ) %>% sum()
  if (nrow(data.frame(loss = loss/nrow(df)) %>% filter(is.na(loss)))){browser()}
  data.frame(loss = loss/nrow(df))
}