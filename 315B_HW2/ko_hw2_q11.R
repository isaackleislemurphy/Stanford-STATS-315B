# STATS 315B Homework 2 Question 11
library(dplyr)
library(gbm)

spam_train_df <- read.csv("/Users/kaiokada/Desktop/Stanford/Q3/STATS315B/spam_stats315B_train.csv", header=FALSE)
spam_test_df <- read.csv("/Users/kaiokada/Desktop/Stanford/Q3/STATS315B/spam_stats315B_test.csv", header=FALSE)
spam_train_df <- data.frame(sapply(spam_train_df, as.numeric))
spam_test_df <- data.frame(sapply(spam_test_df, as.numeric))

rflabs<-c("make", "address", "all", "3d", "our", "over", "remove","internet","order", "mail", "receive", "will","people", "report", "addresses","free", "business","email", "you", "credit", "your", "font","000","money","hp", "hpl", "george", "650", "lab", "labs","telnet", "857", "data", "415", "85", "technology", "1999","parts","pm", "direct", "cs", "meeting", "original", "project","re","edu", "table", "conference", ";", "(", "[", "!", "$", "#","CAPAVE", "CAPMAX", "CAPTOT","type")
colnames(spam_train_df)<-rflabs
colnames(spam_test_df)<-rflabs

# split the dataset
set.seed(131)
train_rows <- sample(1:nrow(spam_train_df), size = as.integer(nrow(spam_train_df) * 0.7)) # 70% training
val_rows <- setdiff(1:nrow(spam_train_df), train_rows)

train_data <- spam_train_df[train_rows,]
val_data <- spam_train_df[val_rows,]

gbm0<-gbm(type~.,data=train_data,train.fraction=0.8,
          interaction.depth=4,shrinkage=.05,
          n.trees=2500,bag.fraction=0.5,
          cv.folds=5,distribution="bernoulli",verbose=T)
best.iter <- gbm.perf(gbm0,method="cv")
gbm0.predict<-predict(gbm0,val_data,type="response",n.trees=best.iter)
thresh <- 0.5
yhat <- sapply(gbm0.predict, FUN=function(x) if (x > thresh) 1 else 0)
ytrue <- val_data$type
misclass <- mean(yhat != ytrue)
misclass

# The estimate of the misclassification rate is 4.78%

gbm0.predict_test<-predict(gbm0,spam_test_df,type="response",n.trees=best.iter)
thresh <- 0.5
yhat_test <- sapply(gbm0.predict_test, FUN=function(x) if (x > thresh) 1 else 0)
ytrue_test <- spam_test_df$type
misclass <- mean(yhat_test != ytrue_test)
misclass
# The actual misclassification rate on the test set is 4.04%

misclass.spam <- mean(yhat_test[which(ytrue_test == 1)] != ytrue_test[which(ytrue_test == 1)])
misclass.spam
# The test set misclassification rate for spam emails is 4.69%

misclass.notspam <- mean(yhat_test[which(ytrue_test == 0)] != ytrue_test[which(ytrue_test == 0)])
misclass.notspam
# The test set misclassification rate for non spam emails is 3.60%

# Want to build spam filter that throws out no more than 0.3% of non-spam emails.

# (Naive threshold approach)
misclass.notspam_series <- c()
misclass.spam_series <- c()
for (thresh in seq(0.98, 0.99, 0.001)) {
  yhat_test <- sapply(gbm0.predict_test, FUN=function(x) if (x > thresh) 1 else 0)
  misclass.spam <- mean(yhat_test[which(ytrue_test == 1)] != ytrue_test[which(ytrue_test == 1)])
  misclass.notspam <- mean(yhat_test[which(ytrue_test == 0)] != ytrue_test[which(ytrue_test == 0)])
  misclass.notspam_series <- c(misclass.notspam_series, misclass.notspam)
  misclass.spam_series <- c(misclass.spam_series, misclass.spam)
}
misclass.spam_series
misclass.notspam_series
# Threshold determined to be 0.988

thresh <- 0.988
yhat_test <- sapply(gbm0.predict_test, FUN=function(x) if (x > thresh) 1 else 0)
ytrue_test <- spam_test_df$type
misclass <- mean(yhat_test != ytrue_test)
misclass
# The overall misclassification of this naive case is 18.12%
misclass.spam <- mean(yhat_test[which(ytrue_test == 1)] != ytrue_test[which(ytrue_test == 1)])
misclass.spam
misclass.notspam <- mean(yhat_test[which(ytrue_test == 0)] != ytrue_test[which(ytrue_test == 0)])
misclass.notspam
# We misclassify 45% of spam emails and only 0.22% of non.spam emails. Clearly
# we should do better.

spam.pred.test <- function(train_data, test_data, weights, thresh, use_best = FALSE) {
  #' @param train_data Data frame used to fit model
  #' @param test_data Data frame used to evaluate model
  #' @param weights Weight values given to training data observations (must have same length)
  #' @param thresh  Threshold used to evaluate output probabilities as spam, not spam
  #' @param use_best Use the optimal number of iterations by gbm.perf (cv)

  gbm0<-gbm(type~.,data=train_data,
            train.fraction=0.8,
            weights=weights,
            interaction.depth=4,shrinkage=.05,
            n.trees=2500,bag.fraction=0.5,
            cv.folds=5,distribution="bernoulli",verbose=T)
  gbm0.predict <- NULL
  if (use_best) {
    best.iter <- gbm.perf(gbm0,method="cv")
    gbm0.predict<-predict(gbm0,test_data,type="response",n.trees=best.iter)
  }
  else {
    gbm0.predict<-predict(gbm0,test_data,type="response",n.trees=300)
  }
  yhat <- sapply(gbm0.predict, FUN=function(x) if (x > thresh) 1 else 0)
  ytrue <- test_data$type
  misclass <- mean(yhat != ytrue)
  misclass.spam <- mean(yhat[which(ytrue == 1)] != ytrue_test[which(ytrue == 1)])
  misclass.notspam <- mean(yhat[which(ytrue == 0)] != ytrue[which(ytrue == 0)])
  list(misclass.overall = misclass, 
       misclass.spam = misclass.spam, 
       misclass.notspam = misclass.notspam)
}

misclass_weighted.overall <- c()
misclass_weighted.notspam <- c()
for (w in seq(100, 600, 100)) {
  print(w)
  weights <- rep(1.0, nrow(spam_train_df))
  weights[which(spam_train_df$type == 0)] = w # upweight non-spam
  weights[which(spam_train_df$type == 1)] = 10
  errors <- spam.pred.test(spam_train_df, spam_test_df, weights, thresh=0.75)
  misclass_w <- errors$misclass.overall
  misclass.notspam_w <- errors$misclass.notspam
  misclass_weighted.overall <- c(misclass_weighted.overall, misclass_w)
  misclass_weighted.notspam <- c(misclass_weighted.notspam, misclass.notspam_w)
}
misclass_weighted.notspam
misclass_weighted.overall 
# We achieve under 0.3% at a threshold of 0.75 and relative weights of 10 for spam, 
# 300-400 for non-spam.
# The overall misclassification rate at this point is only 10%.

weights[which(spam_train_df$type == 0)] = 400 # upweight non-spam
weights[which(spam_train_df$type == 1)] = 10
errors <- spam.pred.test(spam_train_df, spam_test_df, weights, thresh=0.75, use_best=TRUE)
errors$misclass.overall
errors$misclass.spam
errors$misclass.notspam
# Here, we only misclassify 11.73% of the test set. Now, we misclassify only 29% of spam as
# non-spam, and only 0.2% of non-spam as spam.


# Need to determine important features for weighted model
#ytrue <- test_data$age
#mse_test <- mean((yhat_test - ytrue)^2)
#summary(fit.gbm.best,main="RELATIVE INFLUENCE OF ALL PREDICTORS")
