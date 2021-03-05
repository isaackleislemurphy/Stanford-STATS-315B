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

