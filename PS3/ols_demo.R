library(dplyr)
library(glmnet)
source("ingest.R")
source("constants.R")
source("utils.R")

training_data = read.csv("./data/training_data_processed.csv") %>%
  select(-X, -X.1) # comment this out if necessary
  configure_folds()

# then wrap this in a loop/grid of hyperparameters
lapply(1:length(training_data), function(i){
  train_df = training_data[[i]][[1]] %>%
    select_at(c(CONT_COLNAMES, "response"));
  predict_df = training_data[[i]][[2]];
  result = fit_predict(
    train_df=train_df,
    predict_df=predict_df,
    predict_func=predict,
    scale_cols=CONT_COLNAMES,
    formula = as.formula("response~.")
  );
  result
}) -> lm_example

