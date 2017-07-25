
white <- read.csv('~/Desktop/wine_quality/winequality-white.csv', sep=";")
print(str(white))

library(ggplot2)

white$q <- as.factor(white$quality)

ggplot(data=white, aes(x=q)) +
  geom_bar() +
  labs(x = "Wine Quality")


# Load Caret package and partition data into Test and Train

library(caret)
set.seed(507483)

trainIndex <- createDataPartition(white$quality, p = .6, times = 1, list = F)
white$class <- as.factor(ifelse(white$quality > 7, 'good', 'bad'))

wTrain <- white[trainIndex, !(names(white) %in% c('quality','q'))]
wTest <- white[-trainIndex, !(names(white) %in% c('quality','q'))]

str(wTrain)

wFull <- white[, !(names(white) %in% c('quality','q'))]


# Fit an xgboost model - fix learning rate at .1 and find optimum number of trees

library(xgboost)
library(dplyr)

dtrain <- xgb.DMatrix(data=as.matrix(select(wTrain, -class)), 
                      label=sapply(wTrain$class, function(x) ifelse(x=='good', 1, 0)))
dvalidation <- xgb.DMatrix(data=as.matrix(select(wTest, -class)), 
                      label=sapply(wTest$class, function(x) ifelse(x=='good', 1, 0) ))

params = list(eta = .1,
              max_depth = 5,
              gamma = 0, 
              colsample_bytree = .8, 
              min_child_weight = 1, 
              subsample = .5,
              objective = 'binary:logistic',
              eval_metric = 'auc',
              nthread = 4)

xgb <- xgb.train(params = params,
                 data = dtrain,
                 nrounds = 1000,
                 print_every_n = 10L,
                 early_stopping_rounds = 50,
                 maximize = TRUE,
                 watchlist = list(val1 = dvalidation))

#     Stopping. Best iteration:
#     [122]	val1-auc:0.854336


# Fit an axboost model with CV.  Data is a little thin to split many times ...
xgb2 <- xgb.cv(params = params,
                 data = dtrain,
                 nrounds = 1000,
                 print_every_n = 10L,
                 early_stopping_rounds = 50,
                 maximize = TRUE,
                 nfold = 5,
                 stratified = TRUE)



# Use caret to fit an xgboost model, using predefined train and test data
fitControl <- trainControl(index =  replicate(122, trainIndex, FALSE),
                           summaryFunction = twoClassSummary,
                           classProbs = T)

grid <- expand.grid(nrounds = seq(100,200,10),
                    eta = .1,
                    max_depth = 5,
                    gamma = 0, 
                    colsample_bytree = .8, 
                    min_child_weight = 1, 
                    subsample = .8)

gbm_init <- train(class ~ .,
                  nthread = 4,
                  data = wFull,
                  method = 'xgbTree',
                  metric = 'ROC',
                  trControl = fitControl,
                  tuneGrid = grid,
                  verbose = 1)

gbm_init



# CV to select depth, subsample, and min child weight
fitControl2 <- trainControl(method = "repeatedcv",
                           number = 3,
                           repeats = 3,
                           summaryFunction = twoClassSummary,
                           classProbs = T)

grid2 <- expand.grid(nrounds = 122,
                    eta = .1,
                    max_depth = c(2,3,4,5,6),
                    gamma = 0, 
                    colsample_bytree = .8, 
                    min_child_weight = c(1,3,5,10), 
                    subsample = c(.3,.5,.8))

gbm_2 <- train(class ~ .,
                 data = wTrain,
                 method = 'xgbTree',
                 metric = 'ROC',
                 trControl = fitControl2,
                 tuneGrid = grid2,
                 verbose = 1)

gbm_2
plot(gbm_2)




# Run a random parameter search
fitControlR <- trainControl(method = "repeatedcv",
                            number = 3,
                            repeats = 3,
                            summaryFunction = twoClassSummary,
                            classProbs = T,
                            search = "random")

gbmFitR <- train(class ~ .,
                 data = wTrain,
                 method = 'xgbTree',
                 metric = 'ROC',
                 trControl = fitControlR,
                 tuneLength = 30,
                 verbose = 1,
                 nthread = 4)

gbmFitR


# Use adaptive resampling
adaptControl <- trainControl(method = "adaptive_cv",
                             number = 3,
                             repeats = 3,
                             adaptive = list(min = 5, 
                                             alpha = 0.05, 
                                             method = "gls", 
                                             complete = TRUE),
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary,
                             search = "random")

gbmAdapt <- train(class ~ .,
                  data = wTrain,
                  method = 'xgbTree',
                  metric = 'ROC',
                  trControl = adaptControl,
                  tuneLength = 30,
                  verbose = 1,
                  nthread = 4)


# Use bayesian optimization to tune parameters, with random search as starting point

ctrl <- trainControl(method = "repeatedcv",
                     number = 5,
                     repeats = 5,
                     summaryFunction = twoClassSummary,
                     classProbs = T )

xgb_fit_bayes <- function(nroundsIn, etaIn, max_depthIn, gammaIn, colsample_bytreeIn, min_child_weightIn, subsampleIn) {
  txt <- capture.output(
    mod <- train(class ~ .,
                 data = wTrain,
                 method = 'xgbTree',
                 metric = 'ROC',
                 trControl = ctrl,
                 tuneGrid = data.frame(nrounds = nroundsIn, 
                                       eta = etaIn, 
                                       max_depth = max_depthIn, 
                                       gamma = gammaIn, 
                                       colsample_bytree = colsample_bytreeIn, 
                                       min_child_weight = min_child_weightIn, 
                                       subsample = subsampleIn)))
  
  list(Score = getTrainPerf(mod)[, "TrainROC"], Pred = 0)
}

bounds <- list(nroundsIn = c(50L, 1000L), 
               etaIn = c(.0001, .6), 
               max_depthIn = c(1L, 10L), 
               gammaIn = c(1, 10),
               colsample_bytreeIn = c(.1, 1), 
               min_child_weightIn = c(1L, 20L),
               subsampleIn = c(.2, 1))

## Create a grid of values as the input into the BO code
initial_grid <- gbmFitR$results[, c("nrounds", 
                                    "eta", 
                                    "max_depth", 
                                    "gamma", 
                                    "colsample_bytree", 
                                    "min_child_weight", 
                                    "subsample", 
                                    "ROC")]
names(initial_grid) <- c("nroundsIn", 
                         "etaIn", 
                         "max_depthIn", 
                         "gammaIn", 
                         "colsample_bytreeIn", 
                         "min_child_weightIn", 
                         "subsampleIn", 
                         "Value")

initial_grid


library(rBayesianOptimization)


set.seed(8606)
ba_search <- BayesianOptimization(xgb_fit_bayes,
                                  bounds = bounds,
                                  init_grid_dt = initial_grid, 
                                  init_points = 0, 
                                  n_iter = 30,
                                  acq = "ucb", 
                                  kappa = 1, 
                                  eps = 0.0,
                                  verbose = TRUE)


new_grid <- ba_search$History[,-c('Round')]
new_grid



ba_search2 <- BayesianOptimization(xgb_fit_bayes,
                                  bounds = bounds,
                                  init_grid_dt = new_grid, 
                                  init_points = 0, 
                                  n_iter = 10,
                                  acq = "ucb", 
                                  kappa = 1, 
                                  eps = 0.0,
                                  verbose = TRUE)
