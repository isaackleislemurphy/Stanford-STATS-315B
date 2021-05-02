library(dplyr)
library(rpart)
library(rpart.plot)

# ingest data
# data = read.csv("~/Downloads/housetype_stats315B.csv") %>%
#   mutate_all(., as.factor)
CAT_COLNAMES = c("TypeHome", "sex", "MarStat", "Occup", "DualInc", "HouseStat", "Ethnic", "Lang")
data = read.csv("~/Downloads/housetype_stats315B.csv") %>%
  mutate_at(CAT_COLNAMES, as.factor) %>%
  mutate_at(setdiff(colnames(.), CAT_COLNAMES), as.numeric)

# apportion data
set.seed(2020)
train_rows = sample(1:nrow(data), as.integer(.7 * nrow(data)))
val_rows = setdiff(1:nrow(data), train_rows)
dev_rows = sample(1:length(val_rows), as.integer(.5 * length(val_rows)))
test_rows = setdiff(1:length(val_rows), dev_rows)

train_data = data[train_rows, ];
dev_data = data[dev_rows, ]
test_data = data[test_rows, ]

# address new columns in held out data
for (col in colnames(data[, 2:ncol(data)])){
  # if new columns arise in devset, NA them
  tune_support = unique(train_data[, col])
  if (nrow(dev_data[!dev_data[, col] %in% tune_support, ])){
    cat("Adding NA (Dev Set): ", col, "\n")
    dev_data[!dev_data[, col] %in% tune_support, ] = NA
  }
  # same for full prediction
  train_support = unique(bind_rows(train_data, dev_data)[, col])
  
  if (nrow(test_data[!test_data[, col] %in% train_support, ])){
    cat("Adding NA (Test Set): ", col, "\n")
    test_data[!test_data[, col] %in% train_support, ] = NA
  }
}

# build the model formula
model_formula = paste0(
  # do.call("paste0", expand.grid("as.factor(", colnames(data[, 2:ncol(data)]), ")")), 
  do.call("paste0", expand.grid(colnames(data[, 2:ncol(data)]))), 
  collapse = " + "
  )
model_formula = paste0(
  "as.factor(TypeHome) ~ ", model_formula
)

# establish tuning grid of hyperparams
tune_grid = expand.grid(
  maxsurrogate=5, # default is 5
  usesurrogate=2,# default is 2
  cp=c(1e-1, 1e-2, 1e-3), # default is 1e-2
  maxdepth=c(5, 10, 20, 30), # default is 30
  minsplit=c(10, 20, 50, 100) # default is 20
)

# do the tuning
results = rep(NA, nrow(tune_grid))
for (i in nrow(tune_grid)){
  tune_params = tune_grid[i, ]
  model_tune = rpart(
    formula=as.formula(model_formula),
    data=train_data,
    maxsurrogate=tune_params$maxsurrogate,
    usesurrogate=tune_params$usesurrogate,
    cp=tune_params$cp,
    maxdepth=tune_params$maxdepth,
    minsplit=tune_params$minsplit
  )
  yhat = predict(model_tune, newdata=dev_data, type="class")
  results[i] = mean(yhat != dev_data$TypeHome)
}

# extract best tune
best_params = tune_grid[which.min(results), ]
print(best_params)

# fit the model according to best tune
model_fit = rpart(
  formula=as.formula(model_formula),
  data=bind_rows(train_data, dev_data),
  maxsurrogate=best_params$maxsurrogate,
  usesurrogate=best_params$usesurrogate,
  cp=best_params$cp,
  maxdepth=best_params$maxdepth,
  minsplit=best_params$minsplit
)


# make predictions and calculate misclass rate
yhat = predict(model_fit, newdata=test_data, type="class")
cat("Misclassification Error (Test set): ", mean(yhat != test_data$TypeHome))

# plot the result
rpart.plot::prp(model_fit)
