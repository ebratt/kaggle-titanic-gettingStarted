## subject: Kaggle Titanic: Machine Learning from Disaster project
##          https://www.kaggle.com/c/titanic-gettingStarted
## title: "run_analysis.R"
## author: "Eric Bratt"
## date: "Saturday, October 4,2014"
## output: R script
################################################################################
## clear out the environment
rm(list=ls())

################################################################################
## function that checks to see if a package is installed and,if not,installs it
## portions of this code came from http://stackoverflow.com/questions/9341635/how-can-i-check-for-installed-r-packages-before-running-install-packages
load_package <- function(x) {
    if (x %in% rownames(installed.packages())) { print("package already installed...") }
    else { install.packages(x) }
}

################################################################################
# install necessary packages
load_package("lubridate") # easy date-handling
load_package("dplyr")     # data manipulation (ie, joining data frames)
load_package("gclus")     # clustering graphics
load_package("car")       # applied regression analysis
load_package("psych")     # descriptive statistics
load_package("leaps")     # regression subset selection including exhaustive search
load_package("bootstrap") # bootstrap, cross-validation, jackknife
load_package("QuantPsyc") # data screening, testing moderation, mediation and estimating power
load_package("corrgram")  # Calculates correlation of variables and displays the results graphically
load_package("popbio")    # 

library(lubridate)
library(dplyr)
library(gclus)
library(car)
library(psych)
library(leaps)
library(bootstrap)
library(QuantPsyc)
library(corrgram)
library(popbio)

# record date analysis was run
analysis_date <- now()

# load the data
train <- read.csv("./data/train.csv", stringsAsFactors=F)
test  <- read.csv("./data/test.csv", stringsAsFactors=F)
str(train)

# train$Survived <- as.factor(train$Survived)
test$Survived  <- as.factor(0)
train$Pclass   <- as.factor(train$Pclass)
test$Pclass    <- as.factor(test$Pclass)

# make a new factor variable for fare buckets
train$FareBin                                    <- '30+'
train$FareBin[train$Fare < 30 & train$Fare >=20] <- '20-30'
train$FareBin[train$Fare < 20 & train$Fare >=10] <- '10-20'
train$FareBin[train$Fare < 10]                   <- '<10'

test$FareBin                                     <- '30+'
test$FareBin[test$Fare < 30 & test$Fare >=20]    <- '20-30'
test$FareBin[test$Fare < 20 & test$Fare >=10]    <- '10-20'
test$FareBin[test$Fare < 10]                     <- '<10'

# make a new factor variable for titles
titleSplit <- function(x) {
    title <- strsplit(x, split='[,.]')[[1]][2]
    title <- sub(' ', '', title)
    title
}
train$Title <- sapply(train$Name, FUN=titleSplit)
test$Title  <- sapply(test$Name, FUN=titleSplit)

# make high-brow and low-brow factors based on title
setBrow <- function(x) {
    if (x %in% c('Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 
                 'Lady', 'Major', 'Rev', 'Sir', 'the Countess')) 
        'High'
    else
        'Low'
}
train$Brow <- sapply(train$Title, FUN=setBrow)
test$Brow  <- sapply(test$Title, FUN=setBrow)

# check for NA's
table(train$Survived)
table(train$Pclass)
table(test$Pclass)
table(train$Sex)
table(test$Sex)
summary(train$Age) # need to set NA's to mean
summary(test$Age)  # need to set NA's to mean
data <- rbind(test, train)
meanAge <- mean(data$Age, na.rm=T)
train$Age2 <- train$Age
train$Age2[is.na(train$Age)] <- meanAge
test$Age2 <- test$Age
test$Age2[is.na(test$Age)] <- meanAge
table(train$SibSp)
table(test$SibSp)
table(train$Parch)
table(test$Parch)
table(train$FareBin)
table(test$FareBin)
table(train$Title)
table(test$Title)
table(train$Brow)
table(test$Brow)

# plot the logistic function for each independent variable against dependent variable
logi.hist.plot(train$Age2,train$Survived,boxp=FALSE,logi.mod=1,type="hist",col="gray",
               ylabel="Probability",xlabel="Age",mainlabel="Probability of Survival Based on Age")
logi.hist.plot(train$SibSp,train$Survived,boxp=FALSE,logi.mod=1,type="hist",col="gray",
               ylabel="Probability",xlabel="Number of Siblings and Spouse",mainlabel="Probability of Survival Based on Number of Siblings and Spouses")
logi.hist.plot(train$Parch,train$Survived,boxp=FALSE,logi.mod=1,type="hist",col="gray",
               ylabel="Probability",xlabel="Number of Parents and Children",mainlabel="Probability of Survival Based on Number of Parents and Children")
logi.hist.plot(train$Fare,train$Survived,boxp=FALSE,logi.mod=1,type="hist",col="gray",
               ylabel="Probability",xlabel="Fare (USD)",mainlabel="Probability of Survival Based on Ticket Fare (USD)")

## response variable, Survived, is binary, so use logistic regression
# build reduced first-order model with all interaction terms
full <- glm(train$Survived ~ (train$Pclass + train$Sex + train$Age2 + 
                              train$SibSp + train$Parch + train$FareBin + 
                              train$Embarked + train$Title + train$Brow) ^ 2, 
           data=train,
           family=binomial())
summary(full)

# use step() function to run stepwise procedure to remove variables 
# and maximize AIC value
# Stepwise selection: 
step.mod.AIC <- step(full, direction=c("both"), scope = list(upper=full, lower=~1 ), k=2)

full.AIC <- glm(train$Survived ~ train$Pclass + train$Sex + train$Age2 + train$SibSp + 
                train$Parch + train$FareBin + train$Embarked + train$Title + 
                train$Pclass:train$Sex + train$Pclass:train$Age2 + 
                train$Pclass:train$SibSp + train$Pclass:train$Parch + train$Pclass:train$FareBin + 
                train$Pclass:train$Embarked + train$Pclass:train$Title + 
                train$Sex:train$Age2 + train$Sex:train$SibSp + 
                train$Sex:train$Parch + train$Sex:train$FareBin + train$Sex:train$Embarked + 
                train$Sex:train$Title + train$Age2:train$SibSp + 
                train$Age2:train$Parch + train$Age2:train$FareBin + train$Age2:train$Embarked + 
                train$Age2:train$Title + train$Age2:train$Brow + train$SibSp:train$Parch + 
                train$SibSp:train$FareBin + train$SibSp:train$Embarked + 
                train$SibSp:train$Title + train$Parch:train$FareBin + 
                train$Parch:train$Embarked + train$Parch:train$Title +  
                train$FareBin:train$Embarked + train$FareBin:train$Title + 
                train$Embarked:train$Title, 
            data=train,
            family=binomial())
summary(full)

# step-wise regression to minimize BIC
step.mod.BIC <- step(train, direction = "both", scope=list(full, lower=base), k=log(nrow(train)))
summary(step.mod.BIC)

# create ordered correlation plot
train_ints <- train[, c("Age2", "SibSp", "Parch", "Fare")]
train_ints.r=abs(cor(train_ints))
train_ints.col=dmat.color(train_ints.r)
train_ints.o=order.single(train_ints.r)
png('correlations_ordered.png', height=1024, width=2048)
cpairs(train_ints,train_ints.o,panel.colors=train_ints.col,gap=.5, main="Variables Ordered and Colored by Correlation \n(Variables with Highest Correlation are Closest to the Diagonal)")
dev.off()
write.table(cor(train_ints), file="correlation_matrix.csv", sep=",")
# create corrgram
png('corrgram.png', height=1024, width=1024)
corrgram(train_ints, order=TRUE, lower.panel=panel.shade, upper.panel=panel.pie, text.panel=panel.txt, main="Quantitative Variables in PC2/PC1 Order")
dev.off()

# there does not appear to be any multi-colinnearity


## predictions on test data
prediction <- predict(full, test, type="response", se.fit=T)

anova(fit, test="Chisq")      #computes deviance/LR test
confint(fit)                  # 95% CI for the coefficientsexp(coef(fit)) 
                              # compute exp(coefficients) to analyze
                              # change in odds for changes in Xexp(confint(fit)) 
                              # 95% CI for exp(coefficients)
exp(coef(fit))
exp(confint(fit))
predict(fit, type="response") # predicted probability valuesresiduals(fit, type="deviance") 
                              # residuals
residuals(fit, type="deviance") 


# remove non-significant interaction terms from AIC model
full.AIC <- update(full.AIC, .~. - MinivanFlag:HP)
summary(model_rfo_int.AIC)
model_rfo_int.AIC=update(model_rfo_int.AIC, .~. - MinivanFlag:NumCyls)
summary(model_rfo_int.AIC)
model_rfo_int.AIC=update(model_rfo_int.AIC, .~. - SportsFlag:Weight)
summary(model_rfo_int.AIC)
model_rfo_int.AIC=update(model_rfo_int.AIC, .~. - AWDFlag:HP)
summary(model_rfo_int.AIC)
model_rfo_int.AIC=update(model_rfo_int.AIC, .~. - WagonFlag:HP)
summary(model_rfo_int.AIC)
model_rfo_int.AIC=update(model_rfo_int.AIC, .~. - AWDFlag:NumCyls)
summary(model_rfo_int.AIC)
# remove remaining non-significant main terms
model_rfo_int.AIC=update(model_rfo_int.AIC, .~. - WagonFlag)
summary(model_rfo_int.AIC)
write.table(summary(model_rfo_int.AIC)$coefficients, file="model_rfo_int_AIC_coefficients.csv", sep=",")
write.table(anova(model_rfo_int.AIC), file="model_rfo_int_AIC_anova.csv", sep=",")
# step-wise regression to minimize BIC
model_rfo_int.BIC=step(model_rfo_int, direction = "both", scope=list(model_rfo_int, lower=base), k=log(nrow(data)))
summary(model_rfo_int.BIC)
write.table(summary(model_rfo_int.BIC)$coefficients, file="model_rfo_int_BIC_coefficients.csv", sep=",")
write.table(anova(model_rfo_int.BIC), file="model_rfo_int_BIC_anova.csv", sep=",")
#
# MODEL DIAGNOSTICS
# ANALYZE RESIDUALS
#
# LOOK FOR OUTLIERS AND INFLUENTIAL POINTS
# print out only observations that may be influential
write.table(summary(influence.measures(model_rfo_int.BIC)), file="model_rfo_int_BIC_influentials.csv", sep=",")
#
# plot of deleted studentized residuals vs hat values
png('model_rfo_int_BIC_influentials.png')
plot(hatvalues(model_rfo_int.BIC), rstudent(model_rfo_int.BIC))
abline(a=0,b=0, col="red")
# add labels to points
text(hatvalues(model_rfo_int.BIC), rstudent(model_rfo_int.BIC), cex=0.7, pos=2)
dev.off()
#
# residuals histogram
x = rstudent(model_rfo_int.BIC)
png('model_rfo_int_BIC_residuals_hist.png')
hist(x, breaks=100, col="red", xlab="logMSRP (USD)", main="Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="blue", lwd=2)
dev.off()
# 
# plot residuals versus predicted
png('model_rfo_int_BIC_residuals_vs_predicted.png')
plot(fitted(model_rfo_int.BIC), rstandard(model_rfo_int.BIC), main="Predicted vs. Residuals Plot")
abline(a=0, b=0, col="red")
text(fitted(model_rfo_int.BIC), rstandard(model_rfo_int.BIC), cex=0.7, pos=2)
dev.off()
# plot residuals against each independent variable
png('model_rfo_int_BIC_residuals_vs_awdflag.png')
plot(AWDFlag, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
dev.off()
png('model_rfo_int_BIC_residuals_vs_MinivanFlag.png')
plot(MinivanFlag, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
dev.off()
png('model_rfo_int_BIC_residuals_vs_PickupFlag.png')
plot(PickupFlag, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
dev.off()
png('model_rfo_int_BIC_residuals_vs_RWDFlag.png')
plot(RWDFlag, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
dev.off()
png('model_rfo_int_BIC_residuals_vs_SUVFlag.png')
plot(SUVFlag, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
dev.off()
png('model_rfo_int_BIC_residuals_vs_SportsFlag.png')
plot(SportsFlag, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
dev.off()
png('model_rfo_int_BIC_residuals_vs_CMPG.png')
plot(CMPG, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
dev.off()
png('model_rfo_int_BIC_residuals_vs_HP.png')
plot(HP, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
dev.off()
png('model_rfo_int_BIC_residuals_vs_NumCyls.png')
plot(NumCyls, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
dev.off()
png('model_rfo_int_BIC_residuals_vs_weight.png')
plot(Weight, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
dev.off()
#
# matrix of independent variable vs. residuals plots
png('matrix_of_independent_variables_vs_residuals.png', height=1024, width=1024)
layout(matrix(c(1,2,3,4),2,2))
plot(CMPG, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
plot(HP, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
plot(NumCyls, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
plot(Weight, rstudent(model_rfo_int.BIC))
abline(a=0, b=0, col="red")
dev.off()
layout(matrix(c(1),1,1))
# 
# normal probability plot of residuals
png('model_rfo_int_BIC_residuals_qqplot.png')
qqnorm(rstandard(model_rfo_int.BIC))
qqline(rstandard(model_rfo_int.BIC), col="red")
dev.off()
#
lm.beta(model_rfo_int.BIC)
write.table(lm.beta(model_rfo_int.BIC), file="model_rfo_int_BIC_coefficients_standard.csv", sep=",")
#
# MODEL VALIDATION
# 
# K-FOLD TESTING
# m=5 indicates this is a 5-fold cross-validation
# The mean squared error for the 5 folds is displayed
# under the M5 value at the bottom of the output
png('model_rfo_int_BIC_5-fold.png')
results=cv.lm(df=data, model_rfo_int.BIC, m=5, printit=TRUE) # 5-fold cross-validation
dev.off()
write.table(cv.lm(df=data, model_rfo_int.BIC, m=5), file="model_rfo_int_BIC_5-fold.csv", sep=",")
# Another way to compute k-fold cross validation
theta.fit=function(x,y){lsfit(x,y)}
theta.predict=function(model_rfo_int.BIC, x){cbind(1,x)%*%model_rfo_int.BIC$coef}
# matrix of xvariables
X=as.matrix(data[c("AWDFlag", "MinivanFlag", "PickupFlag", "RWDFlag", "SUVFlag", "SportsFlag", "CMPG", "HP", "NumCyls", "Weight")])
# response variable
Y=as.matrix(data[c("MSRP")])
results=crossval(X, Y, theta.fit, theta.predict, ngroup=5)
cor(Y, model_rfo_int.BIC$fitted.values)**2 # model R2
cor(Y, results$cv.fit)**2 # cross-validated R2
#difference:
abs(cor(Y,results$cv.fit)**2-cor(Y, model_rfo_int.BIC$fitted.values)**2)
#
#
# training and testing
select.data=sample(1:nrow(data), 0.75*nrow(data))
# Selecting 75% of the data for training
train.data=data[select.data,]
# Selecting 25% (remaining) of the data for testing purposes
test.data=data[-select.data,]
# Fit selected model on training set for BIC
logMSRP=train.data$MSRP
AWDFlag=train.data$AWDFlag
MinivanFlag=train.data$MinivanFlag
PickupFlag=train.data$PickupFlag
RWDFlag=train.data$RWDFlag
SUVFlag=train.data$SUVFlag
SportsFlag=train.data$SportsFlag
CMPG=train.data$CMPG
HP=train.data$HP
NumCyls=train.data$NumCyls
Weight=train.data$Weight
model_rfo_int_BIC_train=lm(logMSRP ~ AWDFlag + MinivanFlag + PickupFlag + RWDFlag + SUVFlag + SportsFlag + CMPG + HP + NumCyls + Weight + MinivanFlag:Weight + PickupFlag:Weight + RWDFlag:NumCyls + RWDFlag:Weight + SUVFlag:NumCyls + SUVFlag:Weight + HP:Weight, data = train.data)
summary(model_rfo_int_BIC_train) # display results
write.table(summary(model_rfo_int_BIC_train)$coefficients, file="model_rfo_int_BIC_train_coefficients.csv", sep=",")
# Fit selected model on testing set
logMSRP=test.data$MSRP
AWDFlag=test.data$AWDFlag
MinivanFlag=test.data$MinivanFlag
PickupFlag=test.data$PickupFlag
RWDFlag=test.data$RWDFlag
SUVFlag=test.data$SUVFlag
SportsFlag=test.data$SportsFlag
CMPG=test.data$CMPG
HP=test.data$HP
NumCyls=test.data$NumCyls
Weight=test.data$Weight
model_rfo_int_BIC_test=lm(logMSRP ~ AWDFlag + MinivanFlag + PickupFlag + RWDFlag + SUVFlag + SportsFlag + CMPG + HP + NumCyls + Weight + MinivanFlag:Weight + PickupFlag:Weight + RWDFlag:NumCyls + RWDFlag:Weight + SUVFlag:NumCyls + SUVFlag:Weight + HP:Weight, data = test.data)
summary(model_rfo_int_BIC_test) # display results
write.table(summary(model_rfo_int_BIC_test)$coefficients, file="model_rfo_int_BIC_test_coefficients.csv", sep=",")
#
#Create fitted values using test.data data
y_pred=predict.lm(model_rfo_int_BIC_test, test.data)
y_obs=test.data[,"MSRP"]
# Compute RMSE of prediction errors
rmse_m1=sqrt((y_obs - y_pred)%*%(y_obs - y_pred)/nrow(test.data))
rmse_m1
# Compute mean absolute error
mae_m1=(abs(y_obs - y_pred)%*%rep(1, length(y_obs)))/nrow(test.data)
mae_m1
# compute cross-validated R^2_pred
r2_pred = cor(cbind(y_obs,y_pred))**2
r2_train = summary(model_rfo_int_BIC_train)$r.squared
diffr2_m1=abs(r2_train-r2_pred)
#print difference of cross-validate R2 and R2
diffr2_m1[1,2]
#
# PREDICTIONS
# Estimate the average price for automobiles that are out-of-sample observations
# Mitsubishi Lancer LS 4dr ($16,722)
# Ford Excursion 6.8 XLT ($41,475)
# both are missing CMPG and HMPG, so impute with 20.1 and 26.9
#
new=data.frame(SportsFlag=c(0,0), SUVFlag=c(0,1), WagonFlag=c(0,0), MinivanFlag=c(0,0), PickupFlag=c(0,0), AWDFlag=c(0,1), RWDFlag=c(0,0), CMPG=c(20.1,20.1), EngSize=c(2,6.8), HMPG=c(26.9, 26.9), NumCyls=c(4,10), HP=c(225,260), Weight=c(2795, 7190), WheelBase=c(102, 137), Length=c(181, 227), Width=c(67, 80))
new
#
# compute average response value and confidence interval
predict(model_rfo_int.BIC, new, se.fit = T, interval="confidence",level=0.95)
write.table(predict(model_rfo_int.BIC, new, se.fit = T, interval="confidence",level=0.95), file="model_rfo_int_BIC_conf.csv", sep=",")
#compute predictions and prediction intervals
predict(model_rfo_int.BIC, new, se.fit = T, interval="prediction", level=0.95)
write.table(predict(model_rfo_int.BIC, new, se.fit = T, interval="prediction", level=0.95), file="model_rfo_int_BIC_pred.csv", sep=",")
# compute average response value and confidence interval using test
predict(model_rfo_int_BIC_test, new, se.fit = T, interval="confidence",level=0.95)
write.table(predict(model_rfo_int_BIC_test, new, se.fit = T, interval="confidence",level=0.95), file="model_rfo_int_BIC_conf_test.csv", sep=",")
#compute predictions and prediction intervals using test
predict(model_rfo_int_BIC_test, new, se.fit = T, interval="prediction", level=0.95)
write.table(predict(model_rfo_int_BIC_test, new, se.fit = T, interval="prediction", level=0.95), file="model_rfo_int_BIC_pred_test.csv", sep=",")