# load the data
train <- read.csv("./data/train.csv")
test  <- read.csv("./data/test.csv")

# what is the proportion of people who survived in the training data?
prop.table(table(train$Survived))

# add prediction to test data
test$Survived <- rep(0, 418)

# fake submission
submit <- data.frame(PassengerID = test$PassengerId, Survived = test$Survived)
write.csv(submit, file="theyallperish.csv", row.names=F)

## model 1: save women and children first
summary(train$Sex)
prop.table(table(train$Sex, train$Survived),1)

# model 1 submission
test$Survived <- 0
test$Survived[test$Sex == 'femaile'] <- 1

## model 2: look at age
summary(train$Age)

# create a new variable called 'Child'
train$Child <- 0
train$Child[train$Age < 18] <- 1

# figure out how many children/non-children by sex survived
aggregate(Survived ~ Child + Sex, data=train, FUN=sum)
aggregate(Survived ~ Child + Sex, data=train, FUN=length)

avg <- function(x) { sum(x)/length(x) }
aggregate(Survived ~ Child + Sex, data=train, FUN=avg)

## model 3: fares/class
#create bins for < $10, between $10 and $20, between $20 and $30, and more than $30
train$FareBin <- '30+'
train$FareBin[train$Fare < 30 & train$Fare >=20] <- '20-30'
train$FareBin[train$Fare < 20 & train$Fare >=10] <- '10-20'
train$FareBin[train$Fare < 10] <- '<10'

# now look at aggregate again
aggregate(Survived ~ FareBin + Pclass + Sex, data=train, FUN=avg)

# model 3 submission
test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$Fare >= 20] <- 0


## use decision trees
# load the recursive partitioning library
library(rpart)
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")
install.packages("rattle")
install.packages("rpart.plot")
install.packages("RColorBrewer")
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(fit)
Prediction <- predict(fit, test, type="class")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "myfirstdtree.csv", row.names = FALSE)

# change control
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train,
             method="class", control=rpart.control(minsplit=2, cp=0))
fancyRpartPlot(fit)

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train,
             method="class", control=rpart.control( your controls ))
new.fit <- prp(fit,snip=TRUE)$obj
fancyRpartPlot(new.fit)

## feature engineering
train$Name[1]
# combine the data sets
test$Survived <- NA
combi <- rbind(train,test)

# cast name as character to make features
combi$Name <- as.character(combi$Name)
combi$Name[1]
strsplit(combi$Name[1], split='[,.]')
strsplit(combi$Name[1], split='[,.]')[[1]]
strsplit(combi$Name[1], split='[,.]')[[1]][2]

# apply this string split to all titles in the data
mysplit <- function(x) {
  strsplit(x, split='[,.]')[[1]][2]
}
combi$Title <- sapply(combi$Name, FUN=mysplit)
table(combi$Title)
combi$Title <- sub(' ', '', combi$Title)
combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
combi$Title <- factor(combi$Title)
combi$FamilySize <- combi$SibSp + combi$Parch + 1
combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
table(combi$FamilyID)
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)
train <- combi[1:891,]
test <- combi[892:1309,]
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
             data=train, method="class")

## random forests
# bagging and bootstrap aggregation
sample(1:10, replace = TRUE)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])
summary(combi$Age)
summary(combi)
which(combi$Embarked == '')
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)
which(is.na(combi$Fare))
combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)
combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)

install.packages("randomForest")
library(randomForest)
set.seed(415)
train <- combi[1:891,]
test <- combi[892:1309,]
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize +
                      FamilyID2, data=train, importance=TRUE, ntree=2000)
varImpPlot(fit)
# submission for random forest model
Prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE)

## forest of conditional inference trees
install.packages("party")
library(party)
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
               data = train, controls=cforest_unbiased(ntree=2000, mtry=3))
Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "final.csv", row.names = FALSE)
