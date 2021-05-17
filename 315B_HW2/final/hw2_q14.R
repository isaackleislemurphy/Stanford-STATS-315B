# STATS 315B Homework 2 Question 13 and 14(RPART)
library(dplyr)
library(gbm)

occup_df <- read.csv("/Users/kaiokada/Desktop/Stanford/Q3/STATS315B/github/Stanford-STATS-315B/315B_HW2/occup_stats315B.csv", header=FALSE)
occup_df <- sapply(occup_df, as.numeric)
colnames(occup_df) <- c("Occup", "TypeHome", "sex", "MarStat", "age", 
                     "Edu", "Income", "LiveBA", "DualInc", "Persons",
                     "Under18", "HouseStat", "Ethnic", "Lang")
occup_df <- transform(occup_df, 
                    age = as.numeric(age), 
                    Edu = as.numeric(Edu),
                    Income = as.numeric(Income),
                    LiveBA = as.numeric(LiveBA),
                    Persons = as.numeric(Persons),
                    Under18 = as.numeric(Under18),
                    Occup = factor(Occup),
                    TypeHome = factor(TypeHome),
                    sex = factor(sex),
                    MarStat = factor(MarStat),
                    HouseStat = factor(HouseStat),
                    DualInc = factor(DualInc),
                    Ethnic = factor(Ethnic),
                    Lang = factor(Lang))
occup_df
# split the dataset
set.seed(2021)
train_rows <- sample(1:nrow(occup_df), size = as.integer(nrow(age_df) * 0.7)) # 70% training
eval_rows <- setdiff(1:nrow(occup_df), train_rows)
val_rows <- sample(eval_rows, size = as.integer(length(eval_rows) * 0.5)) # 15% dev
test_rows <- setdiff(eval_rows, val_rows)

train_data <- occup_df[train_rows,]
val_data <- occup_df[val_rows,]
test_data <- occup_df[test_rows,]

# Hyperparameter tuning:
tune_grid = expand.grid(
  interaction.depth=c(2, 4, 6),
  shrinkage=c(0.05, 0.10, 0.15)
)
tune_grid2 = expand.grid(
  interaction.depth=c(2),
  shrinkage=c(0.0, 0.005, 0.01)
)

best_grid_row <- 0
min_error = 10000.0

# Fit the tree, find the best parameters
for (i in 1:nrow(tune_grid2)) {
  cat("-")
  set.seed(2021)
  fit.gbm <- gbm(Occup~.,
                 data=train_data,
                 train.fraction=1,
                 interaction.depth=tune_grid2$interaction.depth[i],
                 shrinkage=tune_grid2$shrinkage[i],
                 n.trees=2500,
                 bag.fraction=0.5,
                 cv.folds=5,
                 distribution="multinomial",
                 verbose=F)
  best.iter <- gbm.perf(fit.gbm,method="cv")
  phat <- predict(fit.gbm, val_data, type="response", n.trees=best.iter)
  yhat <- apply(phat, MARGIN=1, FUN=function(x) which.max(x))
  ytrue <- val_data$Occup
  misclass <- mean(yhat != ytrue)
  if (misclass < min_error) {
    min_error = misclass
    best_grid_row = i
  }
}
min_error
tune_grid2[best_grid_row,]

# Apply best hyperparams obtained.
set.seed(2021)
fit.gbm.best <- gbm(Occup~.,data=train_data,
                    train.fraction=1,
                    interaction.depth=tune_grid2$interaction.depth[best_grid_row],
                    shrinkage=tune_grid2$shrinkage[best_grid_row],
                    n.trees=2500,
                    bag.fraction=0.5,
                    cv.folds=5,
                    distribution="multinomial",
                    verbose=T)

best.iter <- gbm.perf(fit.gbm.best, method="cv")
best.iter
phat_test <- predict(fit.gbm.best, test_data, type="response", n.trees=best.iter)
yhat_test <- apply(phat_test, MARGIN=1, FUN=function(x) which.max(x))
ytrue <- test_data$Occup
misclass_test <- mean(yhat_test != ytrue)
misclass_test

# Find misclassification rate per occupation
levels(ytrue)
k_misclass = sapply(levels(ytrue), function(x) mean(yhat_test[ytrue == as.numeric(x)] != as.numeric(x)))
k_misclass
summary(fit.gbm.best,main="RELATIVE INFLUENCE OF ALL PREDICTORS")
