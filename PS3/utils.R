library(dplyr)


# Standard Scaler ---------------------------------------------------------

StandardScaler <- function(){
  list(scale=NULL, is_fit=F, cols=NULL)
}

scaler_fit <- function(scaler, df, cols=CONT_COLNAMES){
  scaler$cols = cols
  scaler$scale = scale(df[, cols], center=T, scale=T)
  scaler$is_fit=T
  scaler
}

scaler_transform <- function(scaler, df){
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
                        predict_func=predict,
                        scale_cols=CONT_COLNAMES, 
                        ...){
  scaler = StandardScaler()
  scaler = scaler_fit(scaler, df=train_df, cols=scale_cols)
  train_df_sc = scaler_transform(scaler, train_df)
  predict_df_sc = scaler_transform(scaler, predict_df)
  fit = model_func(data=train_df_sc, ...)
  predict_df$yhat = predict(fit, newdata=predict_df_sc)
  predict_df
}

