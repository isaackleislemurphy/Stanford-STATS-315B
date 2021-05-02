# STATS 315B Homework Questions 1 and 2 (RPART)
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)

age_df <- read.csv("/Users/kaiokada/Desktop/Stanford/Q3/STATS315B/HW1/age_stats315B.csv")
age_df <- sapply(age_df, as.factor)
age_df <- transform(age_df, 
                    age = as.numeric(age), 
                    Edu = factor(Edu, ordered=TRUE, c(1, 2, 3, 4, 5, 6)),
                    Income = factor(Income, ordered=TRUE, c(1, 2, 3, 4, 5, 6, 7, 8, 9)),
                    LiveBA = factor(LiveBA, ordered=TRUE, c(1, 2, 3, 4, 5)),
                    Persons = factor(Persons, ordered=TRUE, c(1,2,3,4,5,6,7,8,9)),
                    Under18 = factor(Under18, ordered=TRUE, c(0,1,2,3,4,5,6,7,8,9)))
age_df

# split the dataset
set.seed(2020)
train_rows <- sample(1:nrow(age_df), size = as.integer(nrow(age_df) * 0.7)) # 70% training
eval_rows <- setdiff(1:nrow(age_df), train_rows)
val_rows <- sample(eval_rows, size = as.integer(length(eval_rows) * 0.5)) # 15% dev
test_rows <- setdiff(eval_rows, val_rows)

train_data <- age_df[train_rows,]
val_data <- age_df[val_rows,]
test_data <- age_df[test_rows,]

# Hyperparameter tuning:
tune_grid = expand.grid(
  maxsurrogate=c(1, 3, 5), # default is 5
  usesurrogate=c(0, 2),# default is 2
  cp=c(1e-1, 1e-2, 1e-3), # default is 1e-2
  maxdepth=c(5, 10, 20, 30), # default is 30
  minsplit=c(10, 20, 50, 100) # default is 20
)

best_grid_row <- 0
min_error <- 10000

# Fit the tree, find the best parameters
for (i in 1:nrow(tune_grid)) {
  cat("-")
  fit.rpart_large <- rpart(age ~ ., train_data, method = "anova", 
                           control=rpart.control(
                             cp = tune_grid$cp[i],
                             maxsurrogate = tune_grid$maxsurrogate[i],
                             usesurrogate = tune_grid$usesurrogate[i],
                             maxdepth = tune_grid$maxdepth[i],
                             minsplit = tune_grid$minsplit[i]))
  yhat <- rpart.predict(fit.rpart_large, val_data)
  ytrue <- val_data$age
  mse <- mean((yhat - ytrue)^2)
  if (mse < min_error) {
    min_error = mse
    best_grid_row = i
  }
}

# Apply best hyperparams obtained.
fit.rpart_best <- rpart(age ~ ., train_data, method = "anova", 
                        control=rpart.control(
                          cp = tune_grid$cp[best_grid_row],
                          maxsurrogate = tune_grid$maxsurrogate[best_grid_row],
                          usesurrogate = tune_grid$usesurrogate[best_grid_row],
                          maxdepth = tune_grid$maxdepth[best_grid_row],
                          minsplit = tune_grid$minsplit[best_grid_row]))

yhat_test <- rpart.predict(fit.rpart_large, test_data)
ytrue <- test_data$age
mse_test <- mean((yhat_test - ytrue)^2)
tune_grid[best_grid_row,]
plot(fit.rpart_best)
fit.rpart_snipped <- snip.rpart(fit.rpart_best)
rpart.plot(fit.rpart_snipped, type = 3, clip.right.labs = FALSE, branch = .3, under = TRUE)

# Look at specifically optimal CP and number of splits within CP table.
min_xerror_idx <- which.min(fit.rpart_best$cptable[,"xerror"])
fit.rpart_best$cptable[min_xerror_idx,]

plot(fit.rpart_best)
fit.rpart_snipped2 <- snip.rpart(fit.rpart_best)
rpart.plot(fit.rpart_snipped2, type = 3, clip.right.labs = FALSE, branch = .3, under = TRUE)


