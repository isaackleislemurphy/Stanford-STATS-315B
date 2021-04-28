# STATS 315B Homework Questions 1 and 2 (RPART)
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(caret)

age_df <- read.csv("./age_stats315B.csv")
age_df <- sapply(age_df, as.factor)
age_df <- transform(age_df, age = as.numeric(age))

# Recurse back from all splits
fit.rpart_large <- rpart(age ~ ., age_df, method = "anova", control=rpart.control(
  cp = 0.0,
  usesurrogate = 2, 
  xval = 10
))
printcp(fit.rpart_large)
plotcp(fit.rpart_large)
min <- which.min(fit.rpart_large$cptable[, "xerror"])
print(fit.rpart_large$cptable[min,])

fit.rpart_small <- snip.rpart(fit.rpart_large, toss=10)
plot(fit.rpart_small)
rpart.plot(fit.rpart_small)
post(fit.rpart_small, "Regression Tree to Predict Age", filename="rpart_result.ps")