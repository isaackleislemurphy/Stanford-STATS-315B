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
mean(score[[2]]$loss, na.rm = TRUE)
# Best df for spline = 3
# Average loss for November Prediction = 0.2592

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
# Best df for spline = 4, could go higher
# Average loss for November Prediction = 0.17899

# (3) Natural Cubic Spline over all continuous params - Higher DFs ------------
# Interact with county
grid_df3 = expand.grid(
  "param" = c(4:6)
)

# time as an interaction term
score3 <- full.predict.score(grid_df3, dev_data, test_data, model_func2, predict_func, "yhat")
mean(score3[[2]]$loss, na.rm = TRUE)
# Best df for spline = 6, could go higher
# Average loss for November Prediction = 0.1670178

# (4) Natural Cubic Spline over all continuous params - Can we take it higher??so ------------
# Interact with county
grid_df4 = expand.grid(
  "param" = c(7:9)
)

# time as an interaction term
score4 <- full.predict.score(grid_df4, dev_data, test_data, model_func2, predict_func, "yhat")
mean(score4[[2]]$loss, na.rm = TRUE)
# Best df for spline = 9, could go higher!
# Average loss for November Prediction = 0.1501467

# (5) We can try B-Spline Bases
grid_df5 = expand.grid(
  "param" = c(8:10)
)

model_func5 <- function(param, data) {
  formula <- response ~ bs(date_idx, df = param) * (. - date_idx)
  lm(formula, data=data)
}
# Best df for b-spline = 9
# Average loss for November Prediction = 0.1945019 (rank deficient matrix)

score5 <- full.predict.score(grid_df5, dev_data, test_data, model_func5, predict_func, yhat_cols = "yhat")
mean(score5[[2]]$loss, na.rm = TRUE)

# (6) Eliminate some parameters (all Safegraph and FB) so we can fit more knots
grid_df6 = expand.grid(
  "param" = c(10:20)
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
  "param" = c(5:10)
)
model_func8 <- function(param, data) {
  formula <- response ~ . - date_idx + ns(date_idx, df = param) * (county)
  lm(formula, data=data)
}
score8 <- full.predict.score(grid_df8, dev_data, test_data, model_func8, predict_func, yhat_cols = "yhat")
mean(score8[[2]]$loss, na.rm = TRUE)
# Best df for spline = 10, could go higher!
# Average loss for November Prediction = 0.117786


# (9) Explore higher DOFs on this model
grid_df9 = expand.grid(
  "param" = c(15:20)
)

score9 <- full.predict.score(grid_df9, dev_data, test_data, model_func8, predict_func, yhat_cols = "yhat")
mean(score9[[2]]$loss, na.rm = TRUE)
# Best df for spline = 20, could go higher!
# Average loss for November Prediction = 0.1016673

# (10) Try 30 knots
grid_df10 = expand.grid(
  "param" = c(30)
)
score10 <- full.predict.score(grid_df10, dev_data, test_data, model_func8, predict_func, yhat_cols = "yhat")
mean(score10[[2]]$loss, na.rm = TRUE)
# Average loss for November Prediction =0.08086619

# (11) Try Adding Days of the Week
full_df %>% left_join(., data.frame(date_idx = unique(full_df$date_idx), dow = as.character(unique(full_df$date_idx) %% 7)),
          by=c("date_idx")) -> full_df_dow
full_df_dow %>%
  dplyr::select(-X, -X.1) %>% # comment this out if necessary
  configure_folds() -> training_data_dow

dev_data_dow = training_data_dow[DATES_DEV]
test_data_dow = training_data_dow[DATES_HOLDOUT]
test_data_dow

grid_df11 = expand.grid(
  "param" = c(5:10)
)

model_func11 <- function(param, data) {
  formula <- response ~ . - date_idx + ns(date_idx, df = param) * (county + dow)
  lm(formula, data=data)
}

score11 <- full.predict.score(grid_df11, dev_data_dow, test_data_dow, model_func11, predict_func, yhat_cols = "yhat", addl_cols = "dow")
mean(score11[[2]]$loss, na.rm = TRUE)
# Best df based on FV was 8.
# Average loss for November Prediction = 0.1602537

# (12) Don't interact day of week with time smoother
grid_df12 = expand.grid(
  "param" = c(5:10)
)

model_func12 <- function(param, data) {
  formula <- response ~ . - date_idx + ns(date_idx, df = param) * (county)
  lm(formula, data=data)
}

score12 <- full.predict.score(grid_df12, dev_data_dow, test_data_dow, model_func12, predict_func, yhat_cols = "yhat", addl_cols = "dow")
mean(score12[[2]]$loss, na.rm = TRUE)
# Average loss for November Prediction = 0.1180598. Day of week doesn't seem to help much


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
#Holiday indices:
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
# Best df is 8
# Average loss is 0.3329691

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

# (18) Lag another key feature by 7 days
lag_df_holiday %>% cbind(data.frame(lag1sg = lag(lag_df_holiday$safegraph_completely_home_prop, 1))) %>%
  cbind(data.frame(lag2sg = lag(lag_df_holiday$safegraph_completely_home_prop, 2))) %>%
  cbind(data.frame(lag3sg = lag(lag_df_holiday$safegraph_completely_home_prop, 3))) %>%
  cbind(data.frame(lag4sg = lag(lag_df_holiday$safegraph_completely_home_prop, 4))) %>%
  cbind(data.frame(lag5sg = lag(lag_df_holiday$safegraph_completely_home_prop, 5))) %>%
  cbind(data.frame(lag6sg = lag(lag_df_holiday$safegraph_completely_home_prop, 6))) %>%
  cbind(data.frame(lag7sg = lag(lag_df_holiday$safegraph_completely_home_prop, 7))) -> lag2_df_holiday

lag2_df_holiday
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

