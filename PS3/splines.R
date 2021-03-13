library(dplyr)
library(glmnet)
library(glmnetUtils)
library(splines)
source("ingest.R")
source("constants.R")
source("utils.R")
source("fv_util.R")

full_df = read.csv("./data/training_data_processed.csv")
full_df = full_df %>%
left_join(., data.frame(date = unique(full_df$date), date_idx = 1:length(unique(full_df$date))),
             by=c("date"))
 
# Configure cross validation folds for the full dataset
full_df %>%
   dplyr::select(-X, -X.1) %>% # comment this out if necessary
   configure_folds() -> training_data
dev_data = training_data[DATES_DEV]
test_data = training_data[DATES_HOLDOUT]


predict_func <- function(mod, newdata){
  newdata$yhat=stats::predict(mod, newdata=newdata)
  newdata
}

# (1) Natural Cubic Spline over all continuous parameters ------------
# No interaction with county
grid_df = expand.grid(
  "param" = c(1:6)
)

# time as an interaction term
model_func <- function(param, data) {
  formula <- response ~ county + ns(date_idx, df = param) * (. - county - date_idx)
  lm(formula, data=data)
}

score <- full.predict.score(grid_df, dev_data, test_data, model_func, predict_func, "yhat")
sum(score[[2]]["loss"])
# Best df for spline = 3
# Total loss for November Prediction = 7.778277

# (2) Natural Cubic Spline over all continuous params ------------
# Interact with county
grid_df2 = expand.grid(
  "param" = c(1:4)
)
# time as an interaction term

model_func2 <- function(param, data) {
  formula <- response ~ ns(date_idx, df = param) * (. - date_idx)
  lm(formula, data=data)
}

score2 <- full.predict.score(grid_df2, dev_data, test_data, model_func2, predict_func, "yhat")
sum(score2[[2]]["loss"])
# Best df for spline = 4, could go higher
# Total loss for November Prediction = 5.369701

# (3) Natural Cubic Spline over all continuous params - Higher DFs ------------
# Interact with county
grid_df3 = expand.grid(
  "param" = c(4:6)
)

# time as an interaction term
score3 <- full.predict.score(grid_df3, dev_data, test_data, model_func2, predict_func, "yhat")
sum(score3[[2]]["loss"])
# Best df for spline = 6, could go higher
# Total loss for November Prediction = 5.010533

# (4) Natural Cubic Spline over all continuous params - Can we take it higher??so ------------
# Interact with county
grid_df4 = expand.grid(
  "param" = c(7:9)
)
# time as an interaction term
score4 <- full.predict.score(grid_df4, dev_data, test_data, model_func2, predict_func, "yhat")
sum(score4[[2]]["loss"])
# Best df for spline = 9, could go higher!
# Total loss for November Prediction = 4.5044
