setwd("Desktop/AE Final Project/") # change to your working directory

# install libraries
library(caret)
library(ggplot2)
library(caTools)
#unknown
#library(dplyr)
library(lars)
library(glmnet)
#RF
#library(randomForest)
#CART
#library(rpart)
#library(rpart.plot)
#XGBoost
library(xgboost)
library(plyr)
library(mlr)
library(ROCR)
library(DiagrammeR)

set.seed(15071)

data <- read.csv("draft8_reduced.csv")

#convert all the the factor variables to be such
data$CourseTaken = factor(data$CourseTaken)
data$StudentID = factor(data$StudentID)
data$CourseNum = factor(data$CourseNum)
data$MatYear = factor(data$MatYear)
data$Gender = factor(data$Gender)
data$Background = factor(data$Background)
data$PreviousRole = factor(data$PreviousRole)
data$CurrentRole = factor(data$CurrentRole)
#data[,27:87] <- lapply(data[,27:87,drop=FALSE],as.factor) # these are the bag of words cols

data[,27:228] <- scale(data[,27:228])
data <- subset(data, select=-c(decisionmaking, develops, considers, humancomputer, realworld))

str(data) # check to make sure it worked

#split the data to ensure that CourseTaken is equally rep'd
trainIndex <- createDataPartition(data$CourseTaken, p = .8,
                                  list = FALSE,
                                  times = 1)
train <- data[ trainIndex,]
test <- data[-trainIndex,]

# CART CV - not working
cpVals <- data.frame(.cp = seq(0.0001, .003, by=.0001))
Loss <- function(data, lev = NULL, model = NULL, ...) {c(AvgLoss = mean(sum((data$obs - data$pred)^2))) }
trControl <- trainControl(method="cv", number=5, summaryFunction=Loss)
train.cart <- train(train %>% select(-CourseTaken), train$CourseTaken, trControl=trControl, method="rpart", tuneGrid=cpVals, metric="AvgLoss", maximize=FALSE)
courses.cart <- train.cart$finalModel
cp.final = courses.cart$cp

# CART
courses.cart <- rpart(CourseTaken ~ ., data=train, method="class", parms=list(loss=cbind(c(0, 1), c(1, 0))), cp=-Inf)
cart.preds = predict(courses.cart, newdata=test, type = "class")
table(test$CourseTaken, cart.preds)
rpart.plot(courses.cart)

#random forest CV
courseRF.cv = train(y = train$CourseTaken,
                    x = subset(train, select=-c(CourseTaken,CourseNum,StudentID)),
                    method="rf", nodesize=25, ntree=200,
                    trControl=trainControl(method="cv", number=5),
                    tuneGrid=data.frame(mtry=seq(1,51,10)))

plot(courseRF.cv$results$mtry, courseRF.cv$results$RMSE, type = "l")
courseRF.final = courseRF.cv$finalModel

rf.test.preds <- predict(courseRF.final, newdata = data.matrix(subset(test, select=-c(CourseTaken,CourseNum,StudentID,MatYear))))
table(test$CourseTaken, rf.test.preds) # guessing a number that gets us decently close


#XGBoost CV
tg = expand.grid(max_depth = 1:30,
                 eta = .1, 
                 subsample = 0.5, 
                 min_child_weight = 1, 
                 max_delta_step = c(1,2,3,4),
                 gamma = c(0.1, 0.2, 0.3, 0.4, 0.5),  
                 colsample_bytree = 1, 
                 alpha = 0, 
                 lambda = 1)
round.best = rep(0, nrow(tg))
RMSE.cv = rep(0, nrow(tg))
for(i in 1:nrow(tg))
{
  params.new = split(t(tg[i,]), colnames(tg))
  eval = xgb.cv(data = data.matrix(subset(train, select=-c(CourseTaken,CourseNum,StudentID))), label= train$CourseTaken, params = params.new, nrounds = 100, nfold = 5)$evaluation_log$test_rmse_mean
  round.best[i] = which.min(eval)
  RMSE.cv[i] = eval[round.best[i]]
}
winner = which.min(RMSE.cv)

# print out winning parameters and round
tg[winner,]
round.best[winner]

# fill in winning params with hard-coded nums
params.winner = list(max_depth = 3, 
                     eta = 0.1, 
                     subsample = .5, 
                     min_child_weight = 1, 
                     max_delta_step = 1, 
                     gamma = 0.1, 
                     colsample_bytree = 1, 
                     alpha = 0, 
                     lambda = 1)
round.winner = 60

# make the model with the relevant params
mod.xgboost <- xgboost(data = data.matrix(subset(train, select=-c(CourseTaken,CourseNum,StudentID))), label= train$CourseTaken, params = params.winner, nrounds = round.winner, verbose = F)

# make predictions on training set and see the in-sample confusion matrix
xgb.train.preds <- predict(mod.xgboost, newdata = data.matrix(subset(train, select=-c(CourseTaken,CourseNum,StudentID))))
table(train$CourseTaken, xgb.train.preds > 1.25) # guessing a number that gets us decently close

# make predictions on test set and see the OOS confusion matrix
xgb.test.preds <- predict(mod.xgboost, newdata = data.matrix(subset(test, select=-c(CourseTaken,CourseNum,StudentID))))
table(test$CourseTaken, xgb.test.preds > 1.4) # guessing a number that gets us decently close

# plot the ROC
ROCRpred = prediction(xgb.test.preds, test$CourseTaken)
ROCCurve = performance(ROCRpred, "tpr", "fpr")
plot(ROCCurve, colorize=TRUE)
as.numeric(performance(ROCRpred, "auc")@y.values)

#hypothetical iteration
train$XGBPreds <- scale(xgb.train.preds)
test$XGBPreds <- scale(xgb.test.preds)

# plot the first couple of trees
#xgb.plot.tree(model = mod.xgboost, trees = 2:3)
#feature importance
xgb.importance(model = mod.xgboost)

### LASSO ###

all.lambdas <- c(exp(seq(-8, -5, 0.1)))
lasso.cv <- cv.glmnet(x = data.matrix(subset(train, select=-c(CourseTaken,CourseNum,StudentID))), 
                      y = train$CourseTaken, alpha=1, lambda = all.lambdas, nfolds = 5, family = "binomial")
plot(log(lasso.cv$lambda), lasso.cv$cvm)
lasso.lambda.opt <- lasso.cv$lambda.min

idm_lasso_model <- glmnet(x = data.matrix(subset(train, select = -c(CourseTaken,CourseNum,StudentID))),
                    y = train$CourseTaken, lambda = lasso.lambda.opt, alpha = 1, family = "binomial")
idm.preds.l1 <- predict(lasso.mod, newx = data.matrix(subset(test, select=-c(CourseTaken,CourseNum,StudentID))), s=0.001 )  #idm_lasso_model$lambdaOpt


#####

idm_lasso <- expand.grid(alpha = 1,
                         #lambda = 10^(seq(-8, -5, by = 0.1))
                         lambda = lasso.lambda.opt)
idm_cv_lasso <- caret::train(y = train$CourseTaken,
                      x = data.matrix(subset(train, select=-c(CourseTaken,CourseNum,StudentID))),
                      method = "glmnet",
                      trControl = trainControl(method="cv", number=5),
                      tuneGrid = idm_lasso, 
                      family = "binomial")
idm_cv_lasso$results
idm_cv_lasso$bestTune

idm_lasso_model<- idm_cv_lasso$finalModel
idm.preds.l1 <- predict(idm_lasso_model, newx = data.matrix(subset(test, select=-c(CourseTaken,CourseNum,StudentID))), s=0.001 )  #idm_lasso_model$lambdaOpt

# plot the ROC to make sure we're happy
ROCRpred = prediction(idm.preds.l1, test$CourseTaken)
ROCCurve = performance(ROCRpred, "tpr", "fpr")
plot(ROCCurve, colorize=TRUE)
as.numeric(performance(ROCRpred, "auc")@y.values)

#feature importance
lasso_coefs <- coef(idm_lasso_model, s = lasso.lambda.opt)

#prep the output dataframe and save it as csv
ensemble.data <- rbind(train, test)
idm.preds.l1 <- predict(idm_lasso_model, newx = data.matrix(subset(ensemble.data, select=-c(CourseTaken,CourseNum,StudentID))), s=0.001 )  #idm_lasso_model$lambdaOpt
ensemble.data$LassoPreds <- idm.preds.l1
write.csv(ensemble.data, "ensemble_output.csv", row.names=FALSE)

#which courses get the highest scores
df <- data.frame(CourseNum=character(), AvgScore=double(), Deviation=double())
for (x in unique(ensemble.data$CourseNum)){
  temp <- subset(ensemble.data, CourseNum == x)
  avg_score = mean(temp$LassoPreds)
  deviation = sd(temp$LassoPreds)
  df[nrow(df) + 1,] = c(x,avg_score,deviation)
}
df$AvgScore <- sapply(df$AvgScore, as.numeric)
df$Deviation <- sapply(df$Deviation, as.numeric)
coursescores <- df[order(df$AvgScore, decreasing = TRUE),]
write.csv(coursescores, "coursescores.csv", row.names=FALSE)

#which courses get the highest scores for which background
df <- data.frame(Background=character(), CourseNum=character(), AvgScore=double())
for (x in unique(ensemble.data$Background)){
  for (y in unique(ensemble.data$CourseNum)){
    temp <- subset(ensemble.data, Background == x & CourseNum == y)
    avg_score = mean(temp$LassoPreds)
    df[nrow(df) + 1,] = c(x,y,avg_score)
  }
}
df$AvgScore <- sapply(df$AvgScore, as.numeric)
df <- df[order(df$AvgScore, decreasing = TRUE),]
designscores <- subset(df, Background == 'Design')
engineeringscores <- subset(df, Background == 'Engineering')
businessscores <- subset(df, Background == 'Business')
write.csv(designscores, "designscores.csv", row.names=FALSE)
write.csv(engineeringscores, "engineeringscores.csv", row.names=FALSE)
write.csv(businessscores, "businessscores.csv", row.names=FALSE)

#which courses get the highest scores for which gender
df <- data.frame(Gender=character(), CourseNum=character(), AvgScore=double())
for (x in unique(ensemble.data$Gender)){
  for (y in unique(ensemble.data$CourseNum)){
    temp <- subset(ensemble.data, Gender == x & CourseNum == y)
    avg_score = mean(temp$LassoPreds)
    df[nrow(df) + 1,] = c(x,y,avg_score)
  }
}
df$AvgScore <- sapply(df$AvgScore, as.numeric)
df <- df[order(df$AvgScore, decreasing = TRUE),]
malescores <- subset(df, Gender == 'M')
femalescores <- subset(df, Gender == 'F')
write.csv(malescores, "malescores.csv", row.names=FALSE)
write.csv(femalescores, "femalescores.csv", row.names=FALSE)

#which courses have the highest gender disparity
df <- data.frame(CourseNum=character(), MaleScore=double(), FemaleScore=double(), Difference=double())
for (x in unique(ensemble.data$CourseNum)){
  temp_male <- subset(malescores, CourseNum == x)
  temp_female <- subset(femalescores, CourseNum == x)
  difference = temp_male$AvgScore - temp_female$AvgScore
  df[nrow(df) + 1,] = c(x,temp_male$AvgScore,temp_female$AvgScore,difference)
}
df$MaleScore <- sapply(df$MaleScore, as.numeric)
df$FemaleScore <- sapply(df$FemaleScore, as.numeric)
df$Difference <- sapply(df$Difference, as.numeric)
gender.disparity <- df[order(df$Difference, decreasing = TRUE),]
write.csv(gender.disparity, "genderdisparity.csv", row.names=FALSE)

#which courses are popular with people with a given current job
df <- data.frame(CurrentRole=character(), CourseNum=character(), AvgScore=double())
for (x in unique(ensemble.data$CurrentRole)){
  for (y in unique(ensemble.data$CourseNum)){
    temp <- subset(ensemble.data, CurrentRole == x & CourseNum == y)
    avg_score = mean(temp$LassoPreds)
    df[nrow(df) + 1,] = c(x,y,avg_score)
  }
}
df$AvgScore <- sapply(df$AvgScore, as.numeric)
df <- df[order(df$AvgScore, decreasing = TRUE),]
CRstrategistscores <- subset(df, CurrentRole == 'Strategist')
CRentrepreneurscores <- subset(df, CurrentRole == 'Entrepreneur')
CRstudentscores <- subset(df, CurrentRole == 'Current Student')
CRengineerscores <- subset(df, CurrentRole == 'Engineer')
CRdesignresearchscores <- subset(df, CurrentRole == 'Design Research')
CRdesignerscores <- subset(df, CurrentRole == 'Designer')
CRbusinessscores <- subset(df, CurrentRole == 'Business')
CRpmscores <- subset(df, CurrentRole == 'Product Manager')
write.csv(CRstrategistscores, "CRstrategistscores.csv", row.names=FALSE)
write.csv(CRentrepreneurscores, "CRentrepreneurscores.csv", row.names=FALSE)
write.csv(CRstudentscores, "CRstudentscores.csv", row.names=FALSE)
write.csv(CRengineerscores, "CRengineerscores.csv", row.names=FALSE)
write.csv(CRdesignresearchscores, "CRdesignresearchscores.csv", row.names=FALSE)
write.csv(CRdesignerscores, "CRdesignerscores.csv", row.names=FALSE)
write.csv(CRbusinessscores, "CRbusinessscores.csv", row.names=FALSE)
write.csv(CRpmscores, "CRpmscores.csv", row.names=FALSE)


#which courses are popular with people who have more or less experience than average
meanExp = mean(ensemble.data$NbrYearsWorked)

moreExp <- subset(ensemble.data, NbrYearsWorked >= meanExp)
df <- data.frame(CourseNum=character(), AvgScore=double())
for (x in unique(moreExp$CourseNum)){
  temp <- subset(moreExp, CourseNum == x)
  avg_score = mean(temp$LassoPreds)
  df[nrow(df) + 1,] = c(x,avg_score)
}
df$AvgScore <- sapply(df$AvgScore, as.numeric)
moreexperiencescores <- df[order(df$AvgScore, decreasing = TRUE),]
write.csv(moreexperiencescores, "moreexperiencescores.csv", row.names=FALSE)

lessExp <- subset(ensemble.data, NbrYearsWorked < meanExp)
df <- data.frame(CourseNum=character(), AvgScore=double())
for (x in unique(lessExp$CourseNum)){
  temp <- subset(lessExp, CourseNum == x)
  avg_score = mean(temp$LassoPreds)
  df[nrow(df) + 1,] = c(x,avg_score)
}
df$AvgScore <- sapply(df$AvgScore, as.numeric)
lessexperiencescores <- df[order(df$AvgScore, decreasing = TRUE),]
write.csv(lessexperiencescores, "lessexperiencescores.csv", row.names=FALSE)

#which courses have the highest experience disparity
df <- data.frame(CourseNum=character(), MoreExpScore=double(), LessExpScore=double(), Difference=double())
for (x in unique(ensemble.data$CourseNum)){
  temp_more <- subset(moreexperiencescores, CourseNum == x)
  temp_less <- subset(lessexperiencescores, CourseNum == x)
  difference = temp_more$AvgScore - temp_less$AvgScore
  df[nrow(df) + 1,] = c(x,temp_more$AvgScore,temp_less$AvgScore,difference)
}
df$MoreExpScore <- sapply(df$MoreExpScore, as.numeric)
df$LessExpScore <- sapply(df$LessExpScore, as.numeric)
df$Difference <- sapply(df$Difference, as.numeric)
experience.disparity <- df[order(df$Difference, decreasing = TRUE),]
write.csv(experience.disparity, "experiencedisparity.csv", row.names=FALSE)

#which courses are popular for people from which matriculation year
df <- data.frame(MatYear=character(), CourseNum=character(), AvgScore=double())
for (x in unique(ensemble.data$MatYear)){
  for (y in unique(ensemble.data$CourseNum)){
    temp <- subset(ensemble.data, MatYear == x & CourseNum == y)
    avg_score = mean(temp$LassoPreds)
    df[nrow(df) + 1,] = c(x,y,avg_score)
  }
}
df$AvgScore <- sapply(df$AvgScore, as.numeric)
df <- df[order(df$AvgScore, decreasing = TRUE),]
scores.2015 <- subset(df, MatYear == '2015')
scores.2016 <- subset(df, MatYear == '2016')
scores.2017 <- subset(df, MatYear == '2017')
scores.2018 <- subset(df, MatYear == '2018')
scores.2019 <- subset(df, MatYear == '2019')
write.csv(scores.2015, "2015scores.csv", row.names=FALSE)
write.csv(scores.2015, "2016scores.csv", row.names=FALSE)
write.csv(scores.2015, "2017scores.csv", row.names=FALSE)
write.csv(scores.2015, "2018scores.csv", row.names=FALSE)
write.csv(scores.2015, "2019scores.csv", row.names=FALSE)

#which courses have the highest deviation year to year
df <- data.frame(CourseNum=character(), Deviation=double())
for (x in unique(ensemble.data$CourseNum)){
  temp.2015 <- subset(scores.2015, CourseNum == x)
  temp.2016 <- subset(scores.2016, CourseNum == x)
  temp.2017 <- subset(scores.2017, CourseNum == x)
  temp.2018 <- subset(scores.2018, CourseNum == x)
  temp.2019 <- subset(scores.2019, CourseNum == x)
  deviation = sd(c(temp.2015$AvgScore, temp.2016$AvgScore, temp.2017$AvgScore, temp.2018$AvgScore, temp.2019$AvgScore))
  df[nrow(df) + 1,] = c(x,deviation)
}
df$Deviation <- sapply(df$Deviation, as.numeric)
matyear.deviation <- df[order(df$Deviation, decreasing = TRUE),]
write.csv(matyear.deviation, "matyeardeviation.csv", row.names=FALSE)

