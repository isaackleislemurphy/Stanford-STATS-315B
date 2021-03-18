library(dplyr)
library(glmnet)
library(glmnetUtils)
library(splines)
library(ggplot2)
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

which(full_df$response == min(c(full_df$response), na.rm=TRUE))
which(full_df$response == max(c(full_df$response), na.rm=TRUE))
max(full_df$response, na.rm=TRUE)


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
mean(score[[2]]$loss, na.rm = TRUE)
# Best df for spline = 3
# Average loss for November Prediction = 0.5213201

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
mean(score2[[2]]$loss, na.rm = TRUE)
# Best df for spline = 1
# Average loss for November Prediction = 0.3559341

model_func3 <- function(param, data) {
  formula <- response ~ ns(date_idx, df = param) * (.)
  lm(formula, data=data)
}

score3 <- full.predict.score(grid_df2, dev_data, test_data, model_func3, predict_func, "yhat")
mean(score3[[2]]$loss, na.rm = TRUE)
# 0.3950835


# # (3) Natural Cubic Spline over all continuous params - Higher DFs ------------
# # Interact with county
# grid_df3 = expand.grid(
#   "param" = c(4:6)
# )
# 
# # time as an interaction term
# score3 <- full.predict.score(grid_df3, dev_data, test_data, model_func2, predict_func, "yhat")
# mean(score3[[2]]$loss, na.rm = TRUE)
# # Best df for spline = 6, could go higher
# # Average loss for November Prediction = 0.1670178
# 
# # (4) Natural Cubic Spline over all continuous params - Can we take it higher?? ------------
# # Interact with county
# grid_df4 = expand.grid(
#   "param" = c(7:9)
# )
# 
# # time as an interaction term
# score4 <- full.predict.score(grid_df4, dev_data, test_data, model_func2, predict_func, "yhat")
# mean(score4[[2]]$loss, na.rm = TRUE)
# # Best df for spline = 9, could go higher!
# # Average loss for November Prediction = 0.1501467

# (5) We can try B-Spline Bases
grid_df5 = expand.grid(
  "param" = c(4:10)
)

model_func5 <- function(param, data) {
  formula <- response ~ . - date_idx + bs(date_idx, df = param) * (county)
  lm(formula, data=data)
}

score5 <- full.predict.score(grid_df5, dev_data, test_data, model_func5, predict_func, yhat_cols = "yhat")
mean(score5[[2]]$loss, na.rm = TRUE)
# Best df for b-spline = 4
# Average loss for November Prediction = 0.4928848

# (6) Eliminate some parameters (all Safegraph and FB) so we can fit more knots
grid_df6 = expand.grid(
  "param" = c(6:8)
)

model_func6 <- function(param, data) {
  formula <- response ~ ns(date_idx, df = param) * (chng_smoothed_adj_outpatient_cli + chng_smoothed_adj_outpatient_covid + 
    chng_smoothed_outpatient_cli + chng_smoothed_outpatient_covid + 
    doctor.visits_smoothed_adj_cli + doctor.visits_smoothed_cli + 
    hospital.admissions_smoothed_adj_covid19_from_claims + hospital.admissions_smoothed_covid19_from_claims + 
    quidel_covid_ag_smoothed_pct_positive + county - date_idx) 
  mod <- lm(formula, data=data)
  mod
}

score6 <- full.predict.score(grid_df6, dev_data, test_data, model_func6, predict_func, yhat_cols = "yhat")
mean(score6[[2]]$loss, na.rm = TRUE)
# Best df for spline = 20, could go higher!
# Average loss for November Prediction = 0.1209834

# (7) Only the safegraph features
grid_df7 = expand.grid(
  "param" = c(18:20)
)

model_func7 <- function(param, data) {
  formula <- response ~ ns(date_idx, df = param) * (safegraph_bars_visit_prop + safegraph_completely_home_prop + 
                                                      safegraph_completely_home_prop_7dav + safegraph_full_time_work_prop + 
                                                      safegraph_full_time_work_prop_7dav + safegraph_median_home_dwell_time + 
                                                      safegraph_median_home_dwell_time_7dav + safegraph_part_time_work_prop + 
                                                      safegraph_part_time_work_prop_7dav + safegraph_restaurants_visit_num + 
                                                      safegraph_restaurants_visit_prop + county - date_idx) 
  mod <- lm(formula, data=data)
  mod
}

score7 <- full.predict.score(grid_df7, dev_data, test_data, model_func7, predict_func, yhat_cols = "yhat")

# (8) Only interact county with time, regress normally on other params.

grid_df8 = expand.grid(
  "param" = c(5:9)
)
model_func8 <- function(param, data) {
  formula <- response ~ . - date_idx + ns(date_idx, df = param) * (county)
  lm(formula, data=data)
}
score8 <- full.predict.score(grid_df8, dev_data, test_data, model_func8, predict_func, yhat_cols = "yhat")
mean(score8[[2]]$loss, na.rm = TRUE)
# Best df for spline = 8, could go higher!
# Average loss for November Prediction = 0.333344

grid_df8b = expand.grid(
  "param" = c(1:3)
)
model_func8b <- function(param, data) {
  formula <- response ~ . + ns(date_idx, df = param) * (county)
  lm(formula, data=data)
}
score8b <- full.predict.score(grid_df8b, dev_data, test_data, model_func8b, predict_func, yhat_cols = "yhat")
mean(score8b[[2]]$loss, na.rm = TRUE)
# Best df = 1
# Average loss for November Prediction = 0.3674731

# # (9) Explore higher DOFs on this model
# grid_df9 = expand.grid(
#   "param" = c(15:20)
# )
# 
# score9 <- full.predict.score(grid_df9, dev_data, test_data, model_func8, predict_func, yhat_cols = "yhat")
# mean(score9[[2]]$loss, na.rm = TRUE)
# # Best df for spline = 20, could go higher!
# # Average loss for November Prediction = 0.1016673
# 
# # (10) Try 30 knots
# grid_df10 = expand.grid(
#   "param" = c(30)
# )
# score10 <- full.predict.score(grid_df10, dev_data, test_data, model_func8, predict_func, yhat_cols = "yhat")
# mean(score10[[2]]$loss, na.rm = TRUE)
# # Average loss for November Prediction =0.08086619

# (11) Try Adding Days of the Week
full_df %>% left_join(., data.frame(date_idx = unique(full_df$date_idx), dow = as.character(unique(full_df$date_idx) %% 7)),
          by=c("date_idx")) -> full_df_dow
full_df_dow %>%
  dplyr::select(-X, -X.1) %>% # comment this out if necessary
  configure_folds() -> training_data_dow

dev_data_dow = training_data_dow[DATES_DEV]
test_data_dow = training_data_dow[DATES_HOLDOUT]

score11test <- folds.predict(test_data_dow, model_func8, param=8, predict_func, addl_cols = c("dow"))
score11 <- extract_folds_inner(score11test, yhat_cols = "yhat", collapse_func=score_loss)
mean(score11$loss, na.rm = TRUE)

# (11b) Add Index of week
grid_df11b = expand.grid(
  "param" = c(2:4)
)
full_df %>% left_join(., data.frame(date_idx = unique(full_df$date_idx), week_idx = unique(full_df$date_idx) %/% 7),
                      by=c("date_idx")) -> full_df_week
full_df_week %>%
  dplyr::select(-X, -X.1) %>% # comment this out if necessary
  configure_folds() -> training_data_week
training_data_week

dev_data_week = training_data_week[DATES_DEV]
test_data_week = training_data_week[DATES_HOLDOUT]

model_func11b <- function(param, data) {
  formula <- response ~ . - date_idx - week_idx + ns(date_idx, df = 8) * (county) + ns(week_idx, df = param) * (county)
  lm(formula, data=data)
}

score11b <- full.predict.score(grid_df11b, dev_data_week, test_data_week, model_func11b, predict_func, yhat_cols = "yhat", addl_cols = "week_idx")
test_data_dow[1]
mean(score11b[[2]]$loss, na.rm = TRUE)
# Average loss for November Prediction = 0.3340328


# (12) 
grid_df12 = expand.grid(
  "param" = c(8:10)
)

model_func12 <- function(param, data) {
  formula <- response ~ . - date_idx + ns(date_idx, df = param) * (county + dow)
  lm(formula, data=data)
}

score12 <- full.predict.score(grid_df12, dev_data_dow, test_data_dow, model_func12, predict_func, yhat_cols = "yhat", addl_cols = "dow")
mean(score12[[2]]$loss, na.rm = TRUE)
# Average loss for November Prediction = 0.3337217 for df = 8.

# (13) Add spline columns to dataframe
date.spline <- data.frame(ns(full_df$date_idx, df = 30))
spline.cols <- sapply(1:30, function(i) { paste0("spl", i) })
names(date.spline) <- spline.cols
full_df_dow %>% cbind(date.spline) -> spline_df

spline_df %>%
  dplyr::select(-X, -X.1) %>% # comment this out if necessary
  configure_folds() -> training_data_spl

dev_data_spl = training_data_spl[DATES_DEV]
test_data_spl = training_data_spl[DATES_HOLDOUT]

formula.str <- as.formula(paste("response ~ . - date_idx + (", paste(paste(spline.cols, collapse="+" ),  ")* county" )))
model_func13 <- function(data, param) {
  formula <- formula.str
  model <- lm(formula, data=data)
  print(summary(model))
  model
}
grid_df13 = expand.grid(
  "param" = c()
)
score13 <- folds.predict(test_data_spl, model_func13, predict_func, addl_cols = c("dow", spline.cols))
mean(score13[[2]]$loss, na.rm = TRUE)

# (14) Include holiday indicators ------------
# Holiday indices:
full_df %>% mutate(holiday = date %in% c("2020-07-03", "2020-07-04", "2020-07-05", "2020-09-05", 
                                         "2020-09-06", "2020-09-07", "2020-09-08", "2020-11-26", 
                                         "2020-11-27", "2020-11-28", "2020-11-29", "2020-11-30")) -> full_df_holiday

full_df_holiday %>% left_join(., data.frame(date_idx = unique(full_df$date_idx), dow = as.character(unique(full_df$date_idx) %% 7)),
                      by=c("date_idx")) -> full_df_dow_holiday
full_df_dow_holiday %>%
  dplyr::select(-X, -X.1) %>% # comment this out if necessary
  configure_folds() -> training_data_dow_holiday


grid_df14 = expand.grid(
  "param" = c(1)
)

dev_data_holiday = training_data_dow_holiday[DATES_DEV]
test_data_holiday = training_data_dow_holiday[DATES_HOLDOUT]

model_func14 <- function(param, data) {
  formula <- response ~ . - date_idx + ns(date_idx, df = param) * (county)
  lm(formula, data=data)
}

# Compare results agains no holiday
score14test <- folds.predict(test_data_holiday, model_func14, predict_func, param=5, addl_cols = c("dow", "holiday"))
score14 <- extract_folds_inner(score14test, yhat_cols = "yhat", collapse_func=score_loss)
mean(score14$loss, na.rm = TRUE)

score12test <- folds.predict(test_data_dow, model_func12, predict_func, param=5, addl_cols = c("dow"))
score12 <- extract_folds_inner(score12test, yhat_cols = "yhat", collapse_func=score_loss)
mean(score12$loss, na.rm = TRUE)


# (15) More exhaustive parameter search
score15 <- full.predict.score(grid_df15, dev_data_holiday, test_data_holiday, model_func14, predict_func, yhat_cols = "yhat", addl_cols = c("dow", "holiday"))
mean(score15[[2]]$loss, na.rm = TRUE)
print(score15[[2]], n = 30)
score15[1]
grid_df15 = expand.grid(
  "param" = c(5:15)
)
# Best df is 8, Average loss was 0.3329691


# (16) Tiny model
grid_df16 = expand.grid(
  "param" = c(1:10)
)

model_func16 <- function(param, data) {
  formula <- response ~ chng_smoothed_adj_outpatient_cli + fb.survey_smoothed_wnohh_cmnty_cli + 
    fb.survey_smoothed_ili + fb.survey_smoothed_nohh_cmnty_cli + safegraph_completely_home_prop + 
    ns(date_idx, df = param) * (county)
  lm(formula, data=data)
}

score16 <- full.predict.score(grid_df16, dev_data_holiday, test_data_holiday, model_func16, predict_func, yhat_cols = "yhat", addl_cols = c("dow", "holiday"))
mean(score16[[2]]$loss, na.rm = TRUE)

# (17) Lag the first feature by 7 days

full_df_holiday %>% cbind(data.frame(lag1 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 1))) %>%
  cbind(data.frame(lag2 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 2))) %>%
  cbind(data.frame(lag3 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 3))) %>%
  cbind(data.frame(lag4 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 4))) %>%
  cbind(data.frame(lag5 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 5))) %>%
  cbind(data.frame(lag6 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 6))) %>%
  cbind(data.frame(lag7 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 7))) -> lag_df_holiday

lag_df_holiday %>%
  dplyr::select(-X, -X.1) %>% # comment this out if necessary
  configure_folds() -> training_data_lag_holiday
dev_data_lag_holiday = training_data_lag_holiday[DATES_DEV]
test_data_lag_holiday = training_data_lag_holiday[DATES_HOLDOUT]

grid_df17 = expand.grid(
  "param" = c(7:10)
)
score17 <- full.predict.score(grid_df17, dev_data_lag_holiday, test_data_lag_holiday, model_func14, predict_func, yhat_cols = "yhat", addl_cols = c("holiday", "lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7"))
score17[[1]]
mean(score17[[2]]$loss, na.rm = TRUE)
# df 8, average error = 0.3314151

score17test <- folds.predict(test_data_lag_holiday, model_func14, predict_func, param=8, addl_cols = c("holiday", "lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7"))
score17 <- extract_folds_inner(score17test, yhat_cols = "yhat", collapse_func=score_loss)
write.csv(score17)
mean(score17$loss)

# (18) Lag another key feature by 7 days
lag_df_holiday %>% cbind(data.frame(lag1sg = lag(lag_df_holiday$safegraph_completely_home_prop, 1))) %>%
  cbind(data.frame(lag2sg = lag(lag_df_holiday$safegraph_completely_home_prop, 2))) %>%
  cbind(data.frame(lag3sg = lag(lag_df_holiday$safegraph_completely_home_prop, 3))) %>%
  cbind(data.frame(lag4sg = lag(lag_df_holiday$safegraph_completely_home_prop, 4))) %>%
  cbind(data.frame(lag5sg = lag(lag_df_holiday$safegraph_completely_home_prop, 5))) %>%
  cbind(data.frame(lag6sg = lag(lag_df_holiday$safegraph_completely_home_prop, 6))) %>%
  cbind(data.frame(lag7sg = lag(lag_df_holiday$safegraph_completely_home_prop, 7))) -> lag2_df_holiday

lag2_df_holiday %>%
  dplyr::select(-X, -X.1) %>% # comment this out if necessary
  configure_folds() -> training_data_lag2_holiday

dev_data_lag2_holiday = training_data_lag2_holiday[DATES_DEV]
test_data_lag2_holiday = training_data_lag2_holiday[DATES_HOLDOUT]

grid_df18 = expand.grid(
  "param" = c(8:10)
)
score18 <- full.predict.score(grid_df18, dev_data_lag2_holiday, test_data_lag2_holiday, model_func14, predict_func, yhat_cols = "yhat", addl_cols = c("holiday", "lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7", "lag1sg", "lag2sg", "lag3sg", "lag4sg", "lag5sg", "lag6sg", "lag7sg"))
mean(score18[[2]]$loss, na.rm = TRUE)
# 0.3317776, df = 8

# (19) PCA on best model version
training <- test_data_lag_holiday[[1]][[1]]
decomp = prcomp(training[, CONT_COLNAMES[CONT_COLNAMES != "date_idx"]], rank = 15, scale. = T, center = T);
train_df = cbind(data.frame(decomp$x), training[, c("county", "response", "date_idx", "holiday")])
scale_cols=setdiff(colnames(train_df), c("response", "county"))

# use the best tune to make predictions
lapply(1:length(test_data_lag_holiday), function(i){
  # extract training and prediction data for that fold
  cat("Fold ", i, "; ")
  predict_df = cbind(
    predict(object=decomp, newdata = test_data_lag_holiday[[i]][[2]]),
    test_data_lag_holiday[[i]][[2]][, c("county", "response", "date", "date_idx", "holiday")]
  )
  
  # fit and predict for fold
  result = fit_predict(
    train_df=train_df, # fit_and_predict() will scale for us
    predict_df=predict_df, # predict on this (we'll probably just use only the "next" entry, but can't hurt to predict for all)
    # predict using glmnet formula
    model_func=model_func14, # used to fit model
    # predict.glmnet outputs one prediction for each lambda, so each shall be a column
    predict_func = function(fit, newdata){bind_cols(newdata, data.frame(yhat = predict(fit, newdata=newdata)))},
    # we will scale these continuous column names
    scale_cols=setdiff(colnames(train_df), c("response", "county", "holiday", "date")),
    param=8
  ) 
  result %>%
    mutate(fold_idx = i) # for filtering purposes later
}) %>%
  `names<-`(names(test_data_lag_holiday)) -> test_preds # rbind test preds if you wish to compare

# prediction_scoring
pca_test = extract_folds_inner(test_preds, yhat_cols = "yhat", collapse_func=score_loss)
mean(pca_test$loss)


# Assess model 17 on test
test_df = read.csv("./data/test_data_processed.csv")
test_df$response <- NA
training_df = read.csv("./data/training_data_processed.csv")
full_df = rbind(training_df, test_df)
full_df
full_df = full_df %>%
  left_join(., data.frame(date = unique(full_df$date), date_idx = 1:length(unique(full_df$date))),
            by=c("date"))
full_df %>% mutate(holiday = date %in% c("2020-07-03", "2020-07-04", "2020-07-05", "2020-09-05", 
                                         "2020-09-06", "2020-09-07", "2020-09-08", "2020-11-26", 
                                         "2020-11-27", "2020-11-28", "2020-11-29", "2020-11-30",
                                         "2020-12-23", "2020-12-24", "2020-12-25", "2020-12-26",
                                         "2020-12-27", "2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31")) -> full_df_holiday

full_df_holiday %>% cbind(data.frame(lag1 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 1))) %>%
  cbind(data.frame(lag2 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 2))) %>%
  cbind(data.frame(lag3 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 3))) %>%
  cbind(data.frame(lag4 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 4))) %>%
  cbind(data.frame(lag5 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 5))) %>%
  cbind(data.frame(lag6 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 6))) %>%
  cbind(data.frame(lag7 = lag(full_df_holiday$chng_smoothed_adj_outpatient_cli, 7))) -> full_df_lag_holiday

full_df_lag_holiday %>%
  dplyr::select(-X, -X.1) %>% # comment this out if necessary
  configure_folds() -> full_folds

DATES_FULL_TRAIN = c(DATES_DEV, DATES_HOLDOUT)

DATES_TEST = c( 
  "2020-12-01", 
  "2020-12-02", 
  "2020-12-03", 
  "2020-12-04", 
  "2020-12-05", 
  "2020-12-06", 
  "2020-12-07", 
  "2020-12-08", 
  "2020-12-09", 
  "2020-12-10", 
  "2020-12-11", 
  "2020-12-12", 
  "2020-12-13", 
  "2020-12-14", 
  "2020-12-15", 
  "2020-12-16", 
  "2020-12-17", 
  "2020-12-18", 
  "2020-12-19", 
  "2020-12-20", 
  "2020-12-21", 
  "2020-12-22", 
  "2020-12-23", 
  "2020-12-24", 
  "2020-12-25", 
  "2020-12-26", 
  "2020-12-27", 
  "2020-12-28", 
  "2020-12-29", 
  "2020-12-30",
  "2020-12-31"
)

test_df_lag_holiday = full_folds[DATES_TEST]
score17_final <- folds.predict(test_df_lag_holiday, model_func14, predict_func, param=8, addl_cols = c("holiday", "lag1", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7"))
score17_full <- bind_rows(score17_final)
consolidated_predictions <- data.frame(date = score17_full$date, county = score17_full$county, yhat = score17_full$yhat)
length(unique(score17_full$yhat))
score17_full
write.csv(data.frame(date = consolidated_predictions$date, county = consolidated_predictions$county, yhat = consolidated_predictions$yhat), "test_predictions_orig.csv")
test_df %>% merge(consolidated_predictions, by=c("date", "county")) -> aligned_predictions
aligned_predictions[which(aligned_predictions$county == "6019"),]
write.csv(aligned_predictions, "test_set_predictions.csv")
write.csv(data.frame(date = aligned_predictions$date, county = aligned_predictions$county, yhat = aligned_predictions$yhat), "test_predictions_simple.csv")

score17 <- extract_folds_inner(score17test, yhat_cols = "yhat", collapse_func=score_loss)