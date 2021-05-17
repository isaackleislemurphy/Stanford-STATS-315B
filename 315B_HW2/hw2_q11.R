# File for Homework 2 Problem 11


# Loading all packages ----------------------------------------------------

library(dplyr)
library(gbm)
# library(glmnet)
# library(janitor)
# library(MASS)
# library(randomForest)
# library(tidyverse)
# library(tree)



# Reading in the dataset --------------------------------------------------

# Training Data
spam_train_df<- read.csv("spam_stats315B_train.csv", header = FALSE)

# Test Data
spam_test_df<-read.csv("spam_stats315B_test.csv", header=FALSE)



# Pre-processing the data -------------------------------------------------

# variable names vector
rflabs<-c("make", "address", "all", "3d", "our", "over", "remove","internet",
          "order", "mail", "receive", "will","people", "report", "addresses",
          "free", "business","email", "you", "credit", "your", "font","000","money",
          "hp", "hpl", "george", "650", "lab", "labs","telnet", "857", "data", "415", 
          "85", "technology", "1999","parts","pm", "direct", "cs", "meeting", "original", 
          "project","re","edu", "table", "conference", ";", "(", "[", "!", "$", "#","CAPAVE", 
          "CAPMAX", "CAPTOT","type")

# renaming columns in the train and test set
colnames(spam_train)<-rflabs
colnames(spam_test)<-rflabs

# randomizing the training data rows for gbm fitting
set.seed(2021)
train_data<-spam_train[sample(nrow(spam_train)),]




# Fitting the Model -------------------------------------------------------

# Random for bag.fraction
set.seed(444)

gbm0<-gbm(type~.,data=train_data,train.fraction=0.8,interaction.depth=4,
          shrinkage=.05,n.trees=2500,bag.fraction=0.5,cv.folds=5,
          distribution="bernoulli",verbose=T, n.cores=1)



