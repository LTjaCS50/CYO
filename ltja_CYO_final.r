rm(list = ls())
##########################################################
# Create cyo set, validation set (final hold-out test set)
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(gridExtra) 

# Biomechanical features of orthopedic patients:
# Clean datasets from kaggle https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients
# Original datasets from UCI: http://archive.ics.uci.edu/ml/index.php 

# Read file from my Github
url <- "http://github.com/LTjaCS50/CYO/blob/main/column_2C_weka.csv"
patients <- read_csv("https://raw.githubusercontent.com/LTjaCS50/CYO/main/column_2C_weka.csv")
as.data.frame(patients)

# Check that column names are correct
view(patients)

# Check that all of the rows of data are there
dim(patients)

# Validation set will be 10% of patients data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = patients$class, times = 1, p = 0.1, list = FALSE)
cyo <- patients[-test_index,]
temp <- patients[test_index,]
validation_set <- temp

# Make sure the split is correct
dim(cyo)
dim(validation_set)

################################################################
# Create train and test sets from cyo set
################################################################

# Test will be 10% of cyo data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = cyo$class, times = 1, p = 0.1, list = FALSE)
train_set <- cyo[-test_index,]
test_set <- cyo[test_index,]

# Make sure the split is correct
dim(train_set)
dim(test_set)

################################################################
# Understanding the Data Sets
################################################################

# Ratio of normal vs. abnormal in train_set
normal_train <- nrow((train_set %>% filter(class=="Normal")))
abnormal_train <- nrow((train_set %>% filter(class=="Abnormal")))
normal_train / abnormal_train
normal_train / (normal_train + abnormal_train)

# Ratio of normal vs. abnormal in test_set
normal_test <- nrow((test_set %>% filter(class=="Normal")))
abnormal_test <- nrow((test_set %>% filter(class=="Abnormal")))
normal_test / abnormal_test
normal_test / (normal_test + abnormal_test)

# Graph the distribution
ggplot(train_set, aes(class)) + geom_bar(aes(fill = class))

# Are there any NAs?
sum(is.na(patients))

summary(train_set)

view(train_set)

# Box plots of predictors
train_set %>% gather(movement, degree, -class) %>%
  ggplot(aes(class, degree, fill = class)) +
  geom_boxplot() +
  facet_wrap(~movement, scales = "free") +
  theme(axis.text.x = element_blank(), legend.position="bottom")


# Relationship between degree_spondylolisthesis and other predictors
plot1 <- train_set %>% 
  ggplot(aes(degree_spondylolisthesis, pelvic_incidence, color = class)) + 
  geom_point() + theme(axis.text=element_text(size=6),
                       axis.title=element_text(size=6,face="bold"))

plot2 <- train_set %>% 
  ggplot(aes(degree_spondylolisthesis, `pelvic_tilt numeric`, color = class)) + 
  geom_point() + theme(axis.text=element_text(size=6),
                       axis.title=element_text(size=6,face="bold"))

plot3 <- train_set %>% 
  ggplot(aes(degree_spondylolisthesis, pelvic_radius, color = class)) + 
  geom_point() + theme(axis.text=element_text(size=6),
                       axis.title=element_text(size=6,face="bold"))

plot4 <- train_set %>% 
  ggplot(aes(degree_spondylolisthesis, sacral_slope, color = class)) + 
  geom_point() + theme(axis.text=element_text(size=6),
                       axis.title=element_text(size=6,face="bold"))

plot5 <- train_set %>% 
  ggplot(aes(degree_spondylolisthesis, lumbar_lordosis_angle, color = class)) + 
  geom_point() + theme(axis.text=element_text(size=6),
                       axis.title=element_text(size=6,face="bold"))

grid.arrange(plot1, plot2, plot3, plot4, plot5, nrow=5, ncol=1)

# degree_spondylolisthe shows discrepancy between normal and abnormal
# can we just use one predictor?

# model prep

colnames(train_set) <- make.names(colnames(train_set))
colnames(test_set) <- make.names(colnames(test_set))

y_test <- factor(test_set$class)

y_test

# Data preprocessing
nzv <- nearZeroVar(train_set, saveMetrics = TRUE)
nzv

# Model preparation: Cross validation
control <- trainControl(method = "cv", number = 10, p = .9)

############################################################################
# Model 1: Generalized Linear Models
##########################################################################


# Use all of the predictors with the glm model
set.seed(2020)
train_glm1 <- train(class ~ ., 
                    method = "glm", 
                    family="binomial", 
                    trControl = control,
                    data = train_set)
                    
y_hat_glm1 <- predict(train_glm1, test_set, type = "raw")

cm_glm1_acc <- confusionMatrix(y_hat_glm1, y_test)$overall[["Accuracy"]]

cm_glm1_sen <- confusionMatrix(y_hat_glm1, y_test)$byClass[["Sensitivity"]]

cm_glm1_spe <- confusionMatrix(y_hat_glm1, y_test)$byClass[["Specificity"]]

model_glm1 <- tibble(model = "glm - all predictors", 
                     accuracy = cm_glm1_acc,
                     sensitivity = cm_glm1_sen,
                     specificity = cm_glm1_spe)

accuracy_result <- model_glm1
print.data.frame(accuracy_result)

# Use one predictor with the glm model
set.seed(2020)
train_glm2 <- train(class ~ degree_spondylolisthesis, 
                    method = "glm", 
                    family="binomial", 
                    trControl = control,
                    data = train_set)
y_hat_glm2 <- predict(train_glm2, test_set, type = "raw")

cm_glm2_acc <- confusionMatrix(y_hat_glm2, y_test)$overall[["Accuracy"]]

cm_glm2_sen <- confusionMatrix(y_hat_glm2, y_test)$byClass[["Sensitivity"]]

cm_glm2_spe <- confusionMatrix(y_hat_glm2, y_test)$byClass[["Specificity"]]

accuracy_result <- bind_rows(accuracy_result,
                          data_frame(model ="glm - one predictor", 
                                     accuracy = cm_glm2_acc,
                                     sensitivity = cm_glm2_sen,
                                     specificity = cm_glm2_spe))

print.data.frame(accuracy_result)

##################################################################################
# Model 2: k-Nearest Neighbors
##################################################################################

# Use all of the predictors with the knn model
set.seed(2020)

train_knn1 <- train(class ~ ., 
                    method = "knn", 
                    tuneGrid = data.frame(k = seq(1.5, 2.5, 0.1)),
                    trControl = control,
                    data = train_set)
ggplot(train_knn1)

train_knn1$bestTune

y_hat_knn1 <- predict(train_knn1, test_set, type = "raw")

cm_knn1_acc <- confusionMatrix(y_hat_knn1, y_test)$overall[["Accuracy"]]

cm_knn1_sen <- confusionMatrix(y_hat_knn1, y_test)$byClass[["Sensitivity"]]

cm_knn1_spe <- confusionMatrix(y_hat_knn1, y_test)$byClass[["Specificity"]]

accuracy_result <- bind_rows(accuracy_result,
                             data_frame(model ="knn - all predictors", 
                                        accuracy = cm_knn1_acc,
                                        sensitivity = cm_knn1_sen,
                                        specificity = cm_knn1_spe))
                             
print.data.frame(accuracy_result)

#  Use one predictor with the knn model
set.seed(2020)

train_knn2 <- train(class ~ degree_spondylolisthesis, 
                    method = "knn", 
                    tuneGrid = data.frame(k = seq(1.5, 2.5, 0.1)),
                    trControl = control,
                    data = train_set)

ggplot(train_knn2)

train_knn2$bestTune

y_hat_knn2 <- predict(train_knn2, test_set, type = "raw")

cm_knn2_acc <- confusionMatrix(y_hat_knn2, y_test)$overall[["Accuracy"]]

cm_knn2_sen <- confusionMatrix(y_hat_knn2, y_test)$byClass[["Sensitivity"]]

cm_knn2_spe <- confusionMatrix(y_hat_knn2, y_test)$byClass[["Specificity"]]

accuracy_result <- bind_rows(accuracy_result,
                             data_frame(model ="knn - one predictor", 
                                        accuracy = cm_knn2_acc,
                                        sensitivity = cm_knn2_sen,
                                        specificity = cm_knn2_spe))

print.data.frame(accuracy_result)

##################################################################################
# Model 3: Naive Bayes
##################################################################################

# Use all of the predictors with the Naive Bayes model
set.seed(2020)

train_naivebayes1 <- train(class ~ ., 
                           method = "naive_bayes", 
                           trControl = control,
                           data = train_set)

y_hat_naivebayes1 <- predict(train_naivebayes1, test_set, type = "raw")

cm_naivebayes1_acc <- confusionMatrix(y_hat_naivebayes1, y_test)$overall[["Accuracy"]]

cm_naivebayes1_sen <- confusionMatrix(y_hat_naivebayes1, y_test)$byClass[["Sensitivity"]]

cm_naivebayes1_spe <- confusionMatrix(y_hat_naivebayes1, y_test)$byClass[["Specificity"]]

accuracy_result <- bind_rows(accuracy_result,
                             data_frame(model ="naive_bayes - all predictors", 
                                        accuracy = cm_naivebayes1_acc,
                                        sensitivity = cm_naivebayes1_sen,
                                        specificity = cm_naivebayes1_spe))

print.data.frame(accuracy_result)

# Use one predictor with the Naive Bayes model
set.seed(2020)

train_naivebayes2 <- train(class ~ degree_spondylolisthesis, 
                           method = "naive_bayes",
                           trControl = trainControl(method = "cv", number = 10),
                           data = train_set)

y_hat_naivebayes2 <- predict(train_naivebayes2, test_set, type = "raw")

cm_naivebayes2_acc <- confusionMatrix(y_hat_naivebayes2, y_test)$overall[["Accuracy"]]

cm_naivebayes2_sen <- confusionMatrix(y_hat_naivebayes2, y_test)$byClass[["Sensitivity"]]

cm_naivebayes2_spe <- confusionMatrix(y_hat_naivebayes2, y_test)$byClass[["Specificity"]]

accuracy_result <- bind_rows(accuracy_result,
                             data_frame(model ="naive_bayes - one predictor", 
                                        accuracy = cm_naivebayes2_acc,
                                        sensitivity = cm_naivebayes2_sen,
                                        specificity = cm_naivebayes2_spe))

print.data.frame(accuracy_result)


##################################################################################
# Model 4: Decision Tree
##################################################################################

# Use all of the predictors with the Decision Tree model

# A note on using cross validation with rpart: https://stackoverflow.com/questions/33470373/applying-k-fold-cross-validation-model-using-caret-package
# Note: when you fit a tree using rpart, the fitting routine automatically
# performs 10-fold CV and stores the errors for later use 
# (such as for pruning the tree)

set.seed(2020)

train_rpart1 <- train(class ~ ., 
                      method = "rpart", 
                      tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)), 
                      data = train_set)

plot(train_rpart1)

y_hat_rpart1 <- predict(train_rpart1, test_set, type = "raw")

plot(train_rpart1$finalModel, margin = 0.1)
text(train_rpart1$finalModel,  cex = 0.75)

cm_rpart1_acc <- confusionMatrix(y_hat_rpart1, y_test)$overall[["Accuracy"]]

cm_rpart1_sen <- confusionMatrix(y_hat_rpart1, y_test)$byClass[["Sensitivity"]]

cm_rpart1_spe <- confusionMatrix(y_hat_rpart1, y_test)$byClass[["Specificity"]]

accuracy_result <- bind_rows(accuracy_result,
                             data_frame(model ="decision_tree - all predictors", 
                                        accuracy = cm_rpart1_acc,
                                        sensitivity = cm_rpart1_sen,
                                        specificity = cm_rpart1_spe))

print.data.frame(accuracy_result)

# Use one predictor with the Decision Tree model
set.seed(2020)
train_rpart2 <- train(class ~ degree_spondylolisthesis, 
                      method = "rpart", 
                      tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)), 
                      data = train_set)

y_hat_rpart2 <- predict(train_rpart2, test_set, type = "raw")

cm_rpart2_acc <- confusionMatrix(y_hat_rpart2, y_test)$overall[["Accuracy"]]

cm_rpart2_sen <- confusionMatrix(y_hat_rpart2, y_test)$byClass[["Sensitivity"]]

cm_rpart2_spe <- confusionMatrix(y_hat_rpart2, y_test)$byClass[["Specificity"]]

accuracy_result <- bind_rows(accuracy_result,
                             data_frame(model ="decision_tree - one predictor", 
                                        accuracy = cm_rpart2_acc,
                                        sensitivity = cm_rpart2_sen,
                                        specificity = cm_rpart2_spe))

print.data.frame(accuracy_result)

# Segway: addressing identical accuracy between any predictors 
# In this case, between knn all-predictors and decision tree-one predictor

cbind.data.frame(y_hat_knn1, y_hat_rpart2)


#############################################################################
# Model 5: Random Forest
############################################################################

# Use all of the predictors with the Random Forest model
# Optimize node size (per textbook: this is not one of the parameters that 
# the caret package optimizes by default, thus the optimization by the code below).

set.seed(2020)
nodesize <- seq(0.1, 2, 0.1)
train_rf1 <- sapply(nodesize, function(ns){
  train(class ~ ., method = "rf", data = train_set,
        tuneGrid = data.frame(mtry = 2),
        nodesize = ns)$results$Accuracy
})
qplot(nodesize, train_rf1)
n <- nodesize[which.max(train_rf1)]

train_rf1 <- train(class ~ ., method = "rf", nodesize = n, data = train_set)

y_hat_rf1 <- predict(train_rf1, test_set, type = "raw")

cm_rf1_acc <- confusionMatrix(y_hat_rf1, y_test)$overall[["Accuracy"]]

cm_rf1_sen <- confusionMatrix(y_hat_rf1, y_test)$byClass[["Sensitivity"]]

cm_rf1_spe <- confusionMatrix(y_hat_rf1, y_test)$byClass[["Specificity"]]

accuracy_result <- bind_rows(accuracy_result,
                             data_frame(model ="random_forest - all predictors", 
                                        accuracy = cm_rf1_acc,
                                        sensitivity = cm_rf1_sen,
                                        specificity = cm_rf1_spe))

print.data.frame(accuracy_result)

# Use one predictor with the Random Forest model

set.seed(2020)
nodesize2 <- seq(0.1, 2, 0.1)

train_rf2 <- sapply(nodesize2, function(ns){
  train(class ~ degree_spondylolisthesis, method = "rf", data = train_set,
        tuneGrid = data.frame(mtry = 2),
        nodesize = ns)$results$Accuracy
})
qplot(nodesize2, train_rf2)
n2 <- nodesize[which.max(train_rf2)]
n2
train_rf2 <- train(class ~ degree_spondylolisthesis, method = "rf", nodesize = n2, data = train_set)

y_hat_rf2 <- predict(train_rf2, test_set, type = "raw")

cm_rf2_acc <- confusionMatrix(y_hat_rf2, y_test)$overall[["Accuracy"]]

cm_rf2_sen <- confusionMatrix(y_hat_rf2, y_test)$byClass[["Sensitivity"]]

cm_rf2_spe <- confusionMatrix(y_hat_rf2, y_test)$byClass[["Specificity"]]

accuracy_result <- bind_rows(accuracy_result,
                             data_frame(model ="random_forest - one predictor", 
                                        accuracy = cm_rf2_acc,
                                        sensitivity = cm_rf2_sen,
                                        specificity = cm_rf2_spe))

print.data.frame(accuracy_result)

############################################################################
# Ensemble
############################################################################

# Build an ensemble model based on all predictors predictions

a <- as.character(y_hat_glm1)
b <- as.character(y_hat_knn1)
c <- as.character(y_hat_naivebayes1)
d <- as.character(y_hat_rpart1)
e <- as.character(y_hat_rf1)

ensemble_model <- cbind.data.frame(a, b, c, d, e)

ensemble_accuracy <- colMeans(ensemble_model == test_set$class)
ensemble_accuracy
mean(ensemble_accuracy)

votes <- rowMeans(ensemble_model == "Normal")

# if more than half of the models predicted "Normal," then the ensemble's prediction
# is also "Normal." This should be easy to determine since we have an odd number of 
# models.

y_hat_ensemble1 <- ifelse(votes > 0.5, "Normal", "Abnormal")

y_hat_ensemble1 <- as.factor(y_hat_ensemble1)

y_hat_ensemble1

cm_ensemble1_acc <- confusionMatrix(y_hat_ensemble1, y_test)$overall[["Accuracy"]]

cm_ensemble1_sen <- confusionMatrix(y_hat_ensemble1, y_test)$byClass[["Sensitivity"]]

cm_ensemble1_spe <- confusionMatrix(y_hat_ensemble1, y_test)$byClass[["Specificity"]]

accuracy_result <- bind_rows(accuracy_result,
                             data_frame(model ="ensemble - all predictors", 
                                        accuracy = cm_ensemble1_acc,
                                        sensitivity = cm_ensemble1_sen,
                                        specificity = cm_ensemble1_spe))

print.data.frame(accuracy_result)

# Build an ensemble model based on one-predictor predictions

f <- as.character(y_hat_glm2)
g <- as.character(y_hat_knn2)
h <- as.character(y_hat_naivebayes2)
i <- as.character(y_hat_rpart2)
j <- as.character(y_hat_rf2)

ensemble_model2 <- cbind.data.frame(f, g, h, i, j)

ensemble_accuracy2 <- colMeans(ensemble_model2 == test_set$class)
ensemble_accuracy2
mean(ensemble_accuracy2)

votes2 <- rowMeans(ensemble_model2 == "Normal")

# if more than half of the models predicted "Normal," then the ensemble's prediction
# is also "Normal." This should be easy to determine since we have an odd number of 
# models.

y_hat_ensemble2 <- ifelse(votes2 > 0.5, "Normal", "Abnormal")

y_hat_ensemble2 <- as.factor(y_hat_ensemble2)
cm_ensemble2_acc <- confusionMatrix(y_hat_ensemble2, y_test)$overall[["Accuracy"]]

cm_ensemble2_sen <- confusionMatrix(y_hat_ensemble2, y_test)$byClass[["Sensitivity"]]

cm_ensemble2_spe <- confusionMatrix(y_hat_ensemble2, y_test)$byClass[["Specificity"]]

accuracy_result <- bind_rows(accuracy_result,
                             data_frame(model ="ensemble - one predictor", 
                                        accuracy = cm_ensemble2_acc,
                                        sensitivity = cm_ensemble2_sen,
                                        specificity = cm_ensemble2_spe))

print.data.frame(accuracy_result)


############################################################################
# Any data imbalance?
############################################################

# Check decision tree
test_set %>% 
  mutate(y_hatA = y_hat_rpart1) %>%
  group_by(class) %>% 
  summarize(accuracy = mean(y_hatA == class))

# Note: No data imbalance to address. Accuracy for each classification is not affected.

# Var imp
imp <-varImp(train_rpart1)
imp

########################################################################
# Final model against validation set
########################################################################

# Prep
colnames(validation_set) <- make.names(colnames(validation_set))
y_validation <- factor(validation_set$class)

# use train_rpart1, the model that has been trained, against the validation set

y_hat_rpart3 <- predict(train_rpart1, validation_set, type = "raw")

cm_rpart3_acc <- confusionMatrix(y_hat_rpart3, y_validation)$overall[["Accuracy"]]

cm_rpart3_sen <- confusionMatrix(y_hat_rpart3, y_validation)$byClass[["Sensitivity"]]

cm_rpart3_spe <- confusionMatrix(y_hat_rpart3, y_validation)$byClass[["Specificity"]]

validation_result <- tibble(model = "decision_tree_validation - all predictors", 
                     accuracy = cm_rpart3_acc,
                     sensitivity = cm_rpart3_sen,
                     specificity = cm_rpart3_spe)

print.data.frame(validation_result)

# use train_rpart2, the model that has been trained, against the validation set

y_hat_rpart4 <- predict(train_rpart2, validation_set, type = "raw")

cm_rpart4_acc <- confusionMatrix(y_hat_rpart4, y_validation)$overall[["Accuracy"]]

cm_rpart4_sen <- confusionMatrix(y_hat_rpart4, y_validation)$byClass[["Sensitivity"]]

cm_rpart4_spe <- confusionMatrix(y_hat_rpart4, y_validation)$byClass[["Specificity"]]

validation_result <- bind_rows(validation_result,
                            data_frame(model = "decision_tree_validation - one predictor", 
                            accuracy = cm_rpart4_acc,
                            sensitivity = cm_rpart4_sen,
                            specificity = cm_rpart4_spe))

print.data.frame(validation_result)

