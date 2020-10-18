coursera project
================

Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>
(see the section on the Weight Lifting Exercise Dataset).

Data

The training data for this project are available
here:<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available
here:<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

## First we load some important packages

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(rpart)
library(rattle)
```

    ## Loading required package: tibble

    ## Loading required package: bitops

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:rattle':
    ## 
    ##     importance

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(corrplot)
```

    ## corrplot 0.84 loaded

## GETTING DATA

## we load the downloaded data

``` r
trainRaw <- read.csv("traindata.csv")
testRaw <- read.csv("testdata.csv")
dim(trainRaw); dim(testRaw)
```

    ## [1] 19622   160

    ## [1]  20 160

## DATA CLEANING AND PROCESSING

#### As there are many columns so we need to remove unneccessary columns

#### We use near zero variance to remove columns

``` r
NZV <- nearZeroVar(trainRaw, saveMetrics = TRUE)
head(NZV, 20)
```

    ##                        freqRatio percentUnique zeroVar   nzv
    ## X                       1.000000  100.00000000   FALSE FALSE
    ## user_name               1.100679    0.03057792   FALSE FALSE
    ## raw_timestamp_part_1    1.000000    4.26562022   FALSE FALSE
    ## raw_timestamp_part_2    1.000000   85.53154622   FALSE FALSE
    ## cvtd_timestamp          1.000668    0.10192641   FALSE FALSE
    ## new_window             47.330049    0.01019264   FALSE  TRUE
    ## num_window              1.000000    4.37264295   FALSE FALSE
    ## roll_belt               1.101904    6.77810621   FALSE FALSE
    ## pitch_belt              1.036082    9.37722964   FALSE FALSE
    ## yaw_belt                1.058480    9.97349913   FALSE FALSE
    ## total_accel_belt        1.063160    0.14779329   FALSE FALSE
    ## kurtosis_roll_belt   1921.600000    2.02323922   FALSE  TRUE
    ## kurtosis_picth_belt   600.500000    1.61553358   FALSE  TRUE
    ## kurtosis_yaw_belt      47.330049    0.01019264   FALSE  TRUE
    ## skewness_roll_belt   2135.111111    2.01304658   FALSE  TRUE
    ## skewness_roll_belt.1  600.500000    1.72255631   FALSE  TRUE
    ## skewness_yaw_belt      47.330049    0.01019264   FALSE  TRUE
    ## max_roll_belt           1.000000    0.99378249   FALSE FALSE
    ## max_picth_belt          1.538462    0.11211905   FALSE FALSE
    ## max_yaw_belt          640.533333    0.34654979   FALSE  TRUE

``` r
training01 <- trainRaw[, !NZV$nzv]
testing01 <- testRaw[, !NZV$nzv]
dim(training01); dim(testing01)
```

    ## [1] 19622   100

    ## [1]  20 100

#### here reduced dimension shows unnecessary columns are removed

#### there are some columns like username,x,time,etc. we need to remove them

``` r
regex <- grepl("^X|timestamp|user_name", names(training01))
training <- training01[, !regex]
testing <- testing01[, !regex]
```

#### NA’s in data can be a big hurdle.so we remove NA’s too.

``` r
cond <- (colSums(is.na(training)) == 0)  ## we take all colums having 0 NA
training <- training[, cond]
testing <- testing[, cond]
```

#### using correlation we check if variables are correlated or not

``` r
corrplot(cor(training[, -length(names(training))]), method = "color", tl.cex = 0.5)
```

![](pro1_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

## MODEL TRAINING

#### now we subset our training data into training and validation data

``` r
set.seed(56789) # For reproducibile purpose
inTrain <- createDataPartition(training$classe, p = 0.70, list = FALSE)
validation <- training[-inTrain, ]
training <- training[inTrain, ]
dim(training); dim(validation)
```

    ## [1] 13737    54

    ## [1] 5885   54

##### we use set.seed for reproducible purposes.We use 70:30 for training and validation data.

### Prediction using Decision tree

``` r
modelTree <- rpart(classe ~ ., data = training, method = "class")

predictTree <- predict(modelTree, validation, type = "class")
confusionMatrix(validation$classe, predictTree)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1492   37   10   84   51
    ##          B  270  551  120  134   64
    ##          C   55   32  818   49   72
    ##          D  116   17  117  655   59
    ##          E   84   89   61  140  708
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7178          
    ##                  95% CI : (0.7061, 0.7292)
    ##     No Information Rate : 0.3427          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6409          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.7397  0.75895   0.7265   0.6168   0.7421
    ## Specificity            0.9529  0.88602   0.9563   0.9359   0.9242
    ## Pos Pred Value         0.8913  0.48376   0.7973   0.6795   0.6543
    ## Neg Pred Value         0.8753  0.96313   0.9366   0.9173   0.9488
    ## Prevalence             0.3427  0.12336   0.1913   0.1805   0.1621
    ## Detection Rate         0.2535  0.09363   0.1390   0.1113   0.1203
    ## Detection Prevalence   0.2845  0.19354   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.8463  0.82249   0.8414   0.7763   0.8331

##### we use classe as outcome variable and remaining as predictors.Then we use this model to predict values of validation data.confusion matrix shows accuracy of **71.78%** so we should try another algorithm and it is **random forest** as it uses decision tree+bootstraping, error gets reduced

## prediction using Random Forest

``` r
modelRF <- train(classe ~ ., data = training, method = "rf", trControl = trainControl(method = "cv", 5), ntree = 250)
modelRF
```

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    53 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10988, 10990, 10991, 10990, 10989 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9933033  0.9915283
    ##   27    0.9971614  0.9964095
    ##   53    0.9938853  0.9922657
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 27.

``` r
predictRF <- predict(modelRF, validation)
confusionMatrix(validation$classe, predictRF)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    0    0    0    0
    ##          B    1 1137    1    0    0
    ##          C    0    1 1025    0    0
    ##          D    0    0    0  964    0
    ##          E    0    0    0    2 1080
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.9992         
    ##                  95% CI : (0.998, 0.9997)
    ##     No Information Rate : 0.2846         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.9989         
    ##                                          
    ##  Mcnemar's Test P-Value : NA             
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9991   0.9990   0.9979   1.0000
    ## Specificity            1.0000   0.9996   0.9998   1.0000   0.9996
    ## Pos Pred Value         1.0000   0.9982   0.9990   1.0000   0.9982
    ## Neg Pred Value         0.9998   0.9998   0.9998   0.9996   1.0000
    ## Prevalence             0.2846   0.1934   0.1743   0.1641   0.1835
    ## Detection Rate         0.2845   0.1932   0.1742   0.1638   0.1835
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9997   0.9993   0.9994   0.9990   0.9998

#### as this model gives accuracy of **99.89%**,we should opt for this model

### now we can apply this on test data

``` r
predict(modelRF, testing[, -length(names(testing))])
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
