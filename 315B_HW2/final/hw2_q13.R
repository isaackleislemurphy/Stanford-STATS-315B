# STATS 315B Homework 2 Question 13 and 14(RPART)
library(dplyr)
library(gbm)

age_df <- read.csv("/Users/kaiokada/Desktop/Stanford/Q3/STATS315B/HW1/age_stats315B.csv")
age_df <- sapply(age_df, as.numeric)
age_df <- transform(age_df, 
                    age = as.numeric(age), 
                    Edu = factor(Edu, ordered=TRUE, c(1, 2, 3, 4, 5, 6)),
                    Income = factor(Income, ordered=TRUE, c(1, 2, 3, 4, 5, 6, 7, 8, 9)),
                    LiveBA = factor(LiveBA, ordered=TRUE, c(1, 2, 3, 4, 5)),
                    Persons = factor(Persons, ordered=TRUE, c(1,2,3,4,5,6,7,8,9)),
                    Under18 = factor(Under18, ordered=TRUE, c(0,1,2,3,4,5,6,7,8,9)),
                    Occup = factor(Occup),
                    TypeHome = factor(TypeHome),
                    sex = factor(sex),
                    MarStat = factor(MarStat),
                    HouseStat = factor(HouseStat),
                    DualInc = factor(DualInc),
                    Ethnic = factor(Ethnic),
                    Lang = factor(Lang))

                    
age_df
# split the dataset
set.seed(2021)
train_rows <- sample(1:nrow(age_df), size = as.integer(nrow(age_df) * 0.7)) # 70% training
eval_rows <- setdiff(1:nrow(age_df), train_rows)
val_rows <- sample(eval_rows, size = as.integer(length(eval_rows) * 0.5)) # 15% dev
test_rows <- setdiff(eval_rows, val_rows)

train_data <- age_df[train_rows,]
val_data <- age_df[val_rows,]
test_data <- age_df[test_rows,]

# Hyperparameter tuning:
tune_grid = expand.grid(
  interaction.depth=c(2, 4, 6),
  shrinkage=c(0.01, 0.05, 0.10, 0.15)
)
tune_grid2 = expand.grid(
  interaction.depth=c(6, 8, 10),
  shrinkage=c(0.0, 0.005, 0.01)
)

best_grid_row <- 0
min_error = 10000.0

# Fit the tree, find the best parameters
for (i in 1:nrow(tune_grid2)) {
  cat("-")
  set.seed(2021)
  fit.gbm <- gbm(age~.,
                 data=train_data,
                 train.fraction=1,
                 interaction.depth=tune_grid2$interaction.depth[i],
                 shrinkage=tune_grid2$shrinkage[i],
                 n.trees=2500,
                 bag.fraction=0.5,
                 cv.folds=5,
                 distribution="gaussian",
                 verbose=F)
  best.iter <- gbm.perf(fit.gbm,method="cv")
  yhat <- predict(fit.gbm, val_data, type="response", n.trees=best.iter)
  ytrue <- val_data$age
  mse <- mean((yhat - ytrue)^2)
  if (mse < min_error) {
    min_error = mse
    best_grid_row = i
  }
}

tune_grid2[best_grid_row,]

# Apply best hyperparams obtained.
set.seed(2021)
fit.gbm.best <- gbm(age~.,data=train_data,
                    train.fraction=1,
                    interaction.depth=tune_grid2$interaction.depth[best_grid_row],
                    shrinkage=tune_grid2$shrinkage[best_grid_row],
                    n.trees=2500,
                    bag.fraction=0.5,
                    cv.folds=5,
                    distribution="gaussian",
                    verbose=T)

best.iter <- gbm.perf(fit.gbm.best, method="cv")
best.iter
yhat_test <- predict(fit.gbm.best, test_data, type="response", n.trees=best.iter)

ytrue <- test_data$age
mse_test <- mean((yhat_test - ytrue)^2)
summary(fit.gbm.best,main="RELATIVE INFLUENCE OF ALL PREDICTORS")
mse_test
