configure_training_data <- function(filepath="train_data.csv"){
  #' Sets up training data for our use -- NO NEED TO RUN AGAIN
  #' @param filepath: character. The name of the file provided by Prof. Tibshirani

  data_raw = read.csv(filepath)
  unique_counties = sort(unique(data_raw$county))
  saveRDS(unique_counties, "unique_counties.RDS")
  for (county in unique_counties){
    data_raw[, paste0("county_", county)] = as.integer(data_raw$county == county)
  }
  data_raw$date = as.character(data_raw$date)
  write.csv(data_raw, "training_data_processed.csv")
}


configure_folds <- function(df, cv_start=.5){
  #' Sets up the fold indices, so that we're all CV'ing on the same thing
  #' @param df: data.frame. A configured dataframe, having been passed through `configure_training_data()`
  #' @param cv_start: numeric[1]. The point at which to start building folds. E.g. 1/2 would mean dev sets start halfway through time series.
  #' @return : list[str: list[str: data.frame, str: data.frame]]. A nested list of train/dev sets. Each inner list contains the entirety of the 
  #' training set, having been split into train/dev on a particular day. This inner list then contains keys `train` and `dev`, corresponding to the approprite sets
  #' As an example, 'result$`2020-07-04`$train' would return training data PRIOR TO 7/4/2020, while 'result$`2020-07-04`$dev' would return the CV data corresponding to that
  #' particular fold.
  df$date = as.character(df$date)
  dates_all = as.character(sort(unique(df$date)))
  cv_dates = dates_all[dates_all >= dates_all[as.integer(length(dates_all)*cv_start)]]
  folds = lapply(cv_dates, function(dt){
    train = df %>% filter(date < dt);
    dev = df %>% filter(date >= dt)
    list(train=train, dev=dev)
  }) %>%
    `names<-`(cv_dates)
  folds
}


