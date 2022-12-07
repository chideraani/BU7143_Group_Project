rm(list=ls(all=TRUE))

library(class)
library(caret)
library(FNN)
library(dplyr)
library(e1071)
library(gmodels) 
library(psych)
library(gplots)
library(ROSE)
library(neuralnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(fastDummies)

rm(list=ls(all=TRUE))


# K-NN Model

# Reading the Data
data <- read.csv('Student Mental health.csv')


#drop some columns
data <- data[,-c(1,9,10,11)]

#rename columns
colnames(data) <- c('gender','age','course','year','cgpa','marital_status',
                    'depression')


#summary of dataset
summary(data)


#filling missing data in age column
mean_age <- mean(data$age,na.rm=TRUE)
data$age[is.na(data$age)]<- floor(mean_age)


#dealing with categorical data...create a column for each category
course <- dummy_cols(data$course)
year <- dummy_cols(data$year)
cgpa <- dummy_cols(data$cgpa)


#binding the three datasets
data <- cbind(data, course[,-1], year[,-1], cgpa[,-1])


#dropping unneccessary columns
data <- data[, -c(3,4,5)]


# heat-map
heatmap(cor(data))

# annotated heat-map
heatmap.2(cor(data), Rowv = FALSE, Colv = FALSE, dendrogram = "none",
          cellnote = round(cor(data),2),
          notecol = "black", key = FALSE, trace = 'none', margins = c(10,10))


#distribution of the depression column
table(data$depression)



#partition the data...70% training, 30% testing
set.seed(1) 
train.index <- sample(row.names(data), 0.7*dim(data)[1])
valid.index <- setdiff(row.names(data), train.index) 
train.df <- data[train.index,]
valid.df <- data[valid.index,]


# normalize the data
norm.values <- preProcess(train.df, method='range')
train.norm.df <- predict(norm.values, train.df)
valid.norm.df <- predict(norm.values, valid.df)


# optimal k
set.seed(1)
accuracy.df <- data.frame(k = seq(1, 15, 1), sensitivity = rep(0, 15))

for(i in 1:15) {
  
  knn.pred <- class::knn(train = train.norm.df,
                         
                         test = valid.norm.df,
                         
                         cl = train.df$depression, k = i)
  confmat <- confusionMatrix(knn.pred, as.factor(valid.df$depression), positive = "1")
  accuracy.df[i,2] <- confmat$byClass["Sensitivity"]
  
}

accuracy.df

#visualising best k
plot(accuracy.df$k, accuracy.df$sensitivity, type='l', xlab='k',ylab='accuracy')

which(accuracy.df[,2] == max(accuracy.df[,2]))

#fitting with k=5
set.seed(1)
knn.pred <- class::knn(train = train.norm.df, test = valid.norm.df,
                       cl = train.df$depression, k = 5,prob=TRUE)


confusionMatrix(knn.pred, as.factor(valid.df$depression), positive = "1")



#-------------------------------------------------------------------------------




rm(list=ls(all=TRUE))

# NEURAL NETWORKS

nn.df <- read.csv('Student Mental health.csv')


#drop some columns
nn.df <- nn.df[, -c(1,9,10,11)]

#rename columns
colnames(nn.df) <- c('gender','age','course','year','cgpa','marital_status',
                     'depression')


#dealing with categorical data
course <- dummy_cols(nn.df$course)
year <- dummy_cols(nn.df$year)
cgpa <- dummy_cols(nn.df$cgpa)


#binding the two datasets
nn.df <- cbind(nn.df, year[,-1], cgpa[,-1], course[,-1])

#dropping unneccessary rows
nn.df <- nn.df[, -c(3,4,5)]

#filling missing data in age column
mean_age <- mean(nn.df$age,na.rm=TRUE)
nn.df$age[is.na(nn.df$age)]<- floor(mean_age)


#removing redundant categorical variables
nn.df <- nn.df[, -c(8,13,15)]

#partition the data
set.seed(1) 
train.index <- sample(row.names(nn.df), 0.7*dim(nn.df)[1])
valid.index <- setdiff(row.names(nn.df), train.index) 
train.df <- nn.df[train.index,]
valid.df <- nn.df[valid.index,]


#normalisation
norm.values <- preProcess(train.df, method='range')
train.norm.df <- predict(norm.values, train.df)
valid.norm.df <- predict(norm.values, valid.df)


# optimal hidden layers
accuracy.df <- data.frame(layers = seq(1, 15, 1), sensitivity = rep(0, 15), accuracy=rep(0, 15))

for(i in 1:15) {
  set.seed(1)
  nn <- neuralnet(depression ~ ., data = train.norm.df, 
                  linear.output = FALSE, 
                  hidden = i,rep=2,
                  stepmax = 100000)
  
  predict <- compute(nn,valid.norm.df[,-4])
  predict <- predict$net.result
  
  pred1 <- ifelse(predict > 0.5, 1, 0)
  accuracy.df[i,2] <- confusionMatrix(table(pred1, valid.norm.df$depression), positive='1')$byClass["Sensitivity"]
  accuracy.df[i,3] <- confusionMatrix(table(pred1, valid.norm.df$depression), positive='1')$overall[1]
  
  
}

accuracy.df


#neural network with 1 layer & 5 nodes
set.seed(1)
nn <- neuralnet(depression ~ ., data = train.norm.df, rep=2, 
                linear.output = FALSE, hidden = 5,
                stepmax = 100000)

# Prediction on training data
predict <- compute(nn,train.norm.df[,-4])
predict <- predict$net.result

pred1 <- ifelse(predict > 0.5, 1, 0)
confusionMatrix(table(pred1, train.norm.df$depression), positive='1')


# Prediction on validation data
predict <- compute(nn,valid.norm.df[,-4])
predict <- predict$net.result

pred1 <- ifelse(predict > 0.5, 1, 0)
confmat <- confusionMatrix(table(pred1, valid.norm.df$depression), positive='1')
confmat$byClass["Sensitivity"]


#------------------------------------------------------------------------------

rm(list=ls(all=TRUE))

# LOGISTIC REGRESSION

log.df <- read.csv('Student Mental health.csv')


#drop some columns
log.df <- log.df[,-c(1,9,10,11)]

#rename columns
colnames(log.df) <- c('gender','age','course','year','cgpa','marital_status',
                      'depression')


#handling categorical variables
log.df$year <- factor(log.df$year)
log.df$cgpa <- factor(log.df$cgpa)


#dropping unneccessary rows
log.df <- log.df[, -c(3)]

#filling missing data in age column
mean_age <- mean(log.df$age,na.rm=TRUE)
log.df$age[is.na(log.df$age)]<- floor(mean_age)



#partition the data
set.seed(1) 
train.index <- sample(row.names(log.df), 0.7*dim(log.df)[1])
valid.index <- setdiff(row.names(log.df), train.index) 
train.df <- log.df[train.index,]
valid.df <- log.df[valid.index,]



#model building
logit.reg <- glm(depression ~ ., data = train.df, family = "binomial")
options(scipen=999)

summary(logit.reg)

#predictions
logit.reg.pred <- predict(logit.reg, valid.df[, -6], type = "response")

#accuracy
pred_thresh <-  ifelse(logit.reg.pred > 0.5, 1, 0)

confusionMatrix(table(pred_thresh, valid.df$depression), positive = '1')


#-------------------------------------------------------------------------------


rm(list=ls(all=TRUE))

# LOGISTIC REGRESSION with anxiety, panic attack, treatment

log.df2 <- read.csv('Student Mental health.csv')


#drop timestamp
log.df2 <- log.df2[,c(2:11)]

#rename columns
colnames(log.df2) <- c('gender','age','course','year','cgpa','marital_status',
                      'depression','anxiety','panic_attack','treatment')


#handling categorical variables
log.df2$year <- factor(log.df2$year)
log.df2$cgpa <- factor(log.df2$cgpa)

#dropping unneccessary rows
log.df2 <- log.df2[, -c(3)]

#filling missing data in age column
mean_age <- mean(log.df2$age,na.rm=TRUE)
log.df2$age[is.na(log.df2$age)]<- floor(mean_age)

table(log.df2$depression)



#partition the data
set.seed(1) 
train.index <- sample(row.names(log.df2), 0.7*dim(log.df2)[1])
valid.index <- setdiff(row.names(log.df2), train.index) 
train.df <- log.df2[train.index,]
valid.df <- log.df2[valid.index,]



#model building
logit.reg <- glm(depression ~ ., data = train.df, family = "binomial")
options(scipen=999)

summary(logit.reg)

#predictions
logit.reg.pred <- predict(logit.reg, valid.df[, -6], type = "response")

#accuracy
pred_thresh <-  ifelse(logit.reg.pred > 0.5, 1, 0)

confusionMatrix(table(pred_thresh, valid.df$depression), positive = '1')


#-------------------------------------------------------------------------------


# CLASSIFICATION TREES

df <- read.csv('Student Mental health.csv')



#drop some columns
df <- df[, -c(1,9,10,11)]

#rename columns
colnames(df) <- c('gender','age','course','year','cgpa','marital_status',
                  'depression')


#handling categorical variables
df$year <- factor(df$year)
df$cgpa <- factor(df$cgpa)
df$course <- factor(df$course)


#filling missing data in age column
mean_age <- mean(df$age,na.rm=TRUE)
df$age[is.na(df$age)]<- floor(mean_age)



#partition the data
set.seed(1)
train.index <- sample(c(1:dim(df)[1]), dim(df)[1]*0.7)
train.df <- df[train.index, ]
valid.df <- df[-train.index, ]


# default classification tree
default.ct <- rpart(depression ~ ., data = train.df, method = "class")
# plot tree
prp(default.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)


#deeper tree
deeper.ct <- rpart(depression ~ ., data = train.df, method = "class", cp = 0, minsplit = 1)
# count number of leaves
length(deeper.ct$frame$var[deeper.ct$frame$var == "<leaf>"])
# plot tree
prp(deeper.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10,
    box.col=ifelse(deeper.ct$frame$var == "<leaf>", 'gray', 'white'))


# evaluating default tree:
# training set
default.ct.pred.train <- predict(default.ct, train.df, type = "class")
confusionMatrix(default.ct.pred.train, as.factor(train.df$depression),positive='1')
train_sensitivity <- confusionMatrix(default.ct.pred.train, as.factor(train.df$depression),positive='1')$byClass["Sensitivity"]

# validation set
default.ct.pred.valid <- predict(default.ct, valid.df, type = "class")
confusionMatrix(default.ct.pred.valid, as.factor(valid.df$depression),positive='1')
valid_sensitivity <- confusionMatrix(default.ct.pred.valid, as.factor(valid.df$depression),positive='1')$byClass["Sensitivity"]



# evaluating deeper  tree:
# training set
deeper.ct.pred.train <- predict(deeper.ct, train.df, type = "class")
confusionMatrix(deeper.ct.pred.train, as.factor(train.df$depression),positive='1')
train_sensitivity <- confusionMatrix(deeper.ct.pred.train, as.factor(train.df$depression),positive='1')$byClass["Sensitivity"]

# validation set
deeper.ct.pred.valid <- predict(deeper.ct, valid.df, type = "class")
confusionMatrix(deeper.ct.pred.valid, as.factor(valid.df$depression),positive='1')
valid_sensitivity <- confusionMatrix(deeper.ct.pred.valid, as.factor(valid.df$depression),positive='1')$byClass["Sensitivity"]



#using cross-validation
cv.ct <- rpart(depression ~ ., data = train.df, method = "class",
               cp = 0.00001, minsplit = 5, xval = 5)
# use printcp() to print the table.
printcp(cv.ct)


# prunning
pruned.ct <- prune(cv.ct,
                   cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])
length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10)


#training set prediction
pruned.ct.train <- predict(pruned.ct, train.df)
confusionMatrix(as.factor((pruned.ct.train[,1]<0.5)*1), as.factor(train.df$depression), positive='1')
train_sensitivity <- confusionMatrix(as.factor((pruned.ct.train[,1]<0.5)*1), as.factor(train.df$depression), positive='1')$byClass["Sensitivity"]


#validation set prediction
pruned.ct.valid <- predict(pruned.ct, valid.df)
confusionMatrix(as.factor((pruned.ct.valid[,1]<0.5)*1), as.factor(valid.df$depression), positive='1')
valid_sensitivity <- confusionMatrix(as.factor((pruned.ct.valid[,1]<0.5)*1), as.factor(valid.df$depression), positive='1')$byClass["Sensitivity"]



#-------------------------------------------------------------------------------


# RANDOM FOREST
rf <- randomForest(as.factor(depression) ~ ., data = train.df, ntree = 500,
                   mtry = 4, nodesize = 5, importance = TRUE)

## variable importance plot
varImpPlot(rf, type = 1)

# training data
rf.train <- predict(rf, train.df)
confusionMatrix(rf.train, as.factor(train.df$depression), positive='1')

# validation data
rf.valid <- predict(rf, valid.df)
confusionMatrix(rf.valid, as.factor(valid.df$depression), positive='1')


