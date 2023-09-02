#load the dataset
getwd()
setwd("C:Users/duchi/Downloads")

#the dataset had has a header that R doesnt recognise and i dont want it so "header = false" removes the header
data <- read.csv("wdata.csv", header = FALSE)

#Data exploration
View(data)
str(data)
head(data)
tail(data)

#for time series, we can't remove NA, we have to import The imputeTS package in R which provides a set of functions for imputing missing values in time series data. Imputation is the process of filling in missing values with estimated values.
#to interpolate
install.packages("imputeTS")
library(imputeTS)

interpole <- na_interpolation(data)
View(interpole)

#After interpolation, I checked for NA.
sum(is.na(interpole))

#changed the data name
data1 <- interpole
View(data1)

#After checking for NA, there was none. Checked for outliers. Outliers can significantly affect your analysis and models
#Detecting outliers using zcore. Zcore stardadizes the dataset. It is used for outlier detection and data normalization
z_scores <- scale(data1)
threshold <- 3
outliers <- data1[abs(z_scores) > threshold, ]
View(outliers)
print(outliers)

class(data1)
IQR_value <- apply (data1, 2, IQR)
Q1 <- apply(data1,2,quantile,0.25)
Q3 <- apply(data1,2,quantile,0.75)
lower_bound<- Q1-1.5*IQR_value
upper_bound <- Q3+1.5*IQR_value
IQR_outliers <- data1[data1<lower_bound|data1>upper_bound]
print(IQR_outliers)

#We have seen that there are outliers, we have to handle those outliers. 
#winsorization replaces extreme values with less extreme values, this is better than removing the outliers. This helps the dataset retain valuable information and is representative of the wider data
#we import the desctools library which helps to handle outliers
install.packages("DescTools")
library(DescTools)
winsorized_data <- apply(data1,2,Winsorize,probs=c(0.05,0.95))
winsorized_data
print(winsorized_data)
View(winsorized_data)

data2 <- winsorized_data
View(data2)

install.packages("tidyverse")
install.packages("lubridate")
install.packages("forecast")
install.packages("ggplot2")
install.packages("gridExtra")
install.packages("caret")
install.packages("lattice")
install.packages("dplyr")

library(tidyverse)
library(lubridate)
library(forecast)
library(ggplot2)
library(gridExtra)
library(caret)
library(lattice)
library(dplyr)

write.csv(data2, file = "data2.csv", row.names = FALSE)

#Importing cleaned data and and Selected Header.
clean_data<- read.csv("data2.csv", header = FALSE)
View(clean_data)
header <- read.csv("tdata.csv", header = TRUE)

mdata <- rbind(header, clean_data)
View(mdata)

write.csv(mdata, file = "mdata.csv", row.names = FALSE)

new_data<- read.csv("mdata.csv", header = TRUE)
View(new_data)

#select row for latitude of east anglia
selected_row <- new_data[154, ]

#extract soil moisture columns for that row
smois_columns <- grep("SMOIS", names(selected_row), ignore.case = TRUE)
smois_values <- unlist(selected_row[smois_columns])

# Define the starting date and time
start_date <- as.POSIXct("2018-05-01 00:00:00", tz = "UTC")

# Calculate the total number of SMOIS measurements in the selected_row
smois_numbers <- length(smois_values)

# Generate a sequence of 3-hour intervals
time_intervals <- seq(from = start_date, by = "3 hours", length.out = smois_numbers)

# Create a new data frame with date and time intervals and SMOIS values
data_table <- data.frame(DateTime = time_intervals, SMOIS = smois_values)
#Create time stamp column
data_table$TimeStamp <- seq_len(nrow(data_table))
View(data_table)


#Visualizations for Soil Moisture over time line plots
ggplot(data_table, aes(x = DateTime, y = SMOIS)) +
  geom_line() +
  labs(title = "Soil Moisture (SMOIS) Over Time",
       x = "Date and Time",
       y = "Soil Moisture (SMOIS)") +
  theme_minimal()

#Visualizations for scatter plot
ggplot(data_table, aes(x = DateTime, y = SMOIS)) +
  geom_point() +
  labs(title = "Soil Moisture (SMOIS) Over Time",
       x = "Date and Time",
       y = "Soil Moisture (SMOIS)") +
  theme_minimal()

#Timeseries with smoothen/rolling average
ggplot(data_table, aes(x = DateTime, y = SMOIS)) +
  geom_line() +
  geom_smooth(method = "loess", span = 0.1, se = FALSE,color = "orange")
  labs(title = "Soil Moisture (SMOIS) Over Time",
       x = "Date and Time",
       y = "Soil Moisture (SMOIS)") +
  theme_minimal()

#MODELLING
#DIVIDE DATA INTO TRAIN AND TEST and round it down to the number of rows
n_rows <- nrow(data_table)
train_size <- floor (0.8* n_rows)

#Set a seed for reproducibility
set.seed(123)
index <- sample(seq_len(n_rows), size = train_size)
train_set <- data_table[index,]
test_set <- data_table[-index,]

#ARIMA MODELLING
#Fitting an ARIMA model to the Trainset Data (SMOIS Column)
arima_model <- auto.arima(train_set$SMOIS, seasonal = TRUE, stepwise = TRUE)

#Forecast Values for Test Set
arima_forecast <- forecast(arima_model, h = length(train_set$SMOIS))
arima_model
arima_forecast

#Calculate the RMSE on test set
rmse_arima <- sqrt(mean((test_set$SMOIS - arima_forecast$mean)^2))

#Print Root Mean Square Error
cat("ARIMA RMSE:", rmse_arima)


#Linear Regression
train_model <- lm(SMOIS ~ TimeStamp, data = train_set)
summary(train_model)

# Predict TSK values for the test set
predictions <- predict(train_model, ndata = test_set)

# Calculate the root mean squared error (RMSE)
LINEAR_REGRESSION_RMSE<- sqrt(mean(test_set$SMOIS - predictions)^2)
cat("RMSE LINEAR:", LINEAR_REGRESSION_RMSE)


# Install and load the required package
install.packages("e1071")
library(e1071)

# Fit an SVR model on the training set with radial kernel
svr_model_radial <- svm(SMOIS~TimeStamp, data = train_set, kernel = "radial")

# Predict TSK values for the test set using the radial kernel SVR model

svr_predictions_radial <- predict(svr_model_radial, ndata = test_set)

# Calculate the root mean squared error (RMSE) for the radial kernel SVR model

svr_rmse_radial <- sqrt(mean((test_set$SMOIS - svr_predictions_radial)^2))
cat("SVR_RADIAL RMSE:", svr_rmse_radial)



# Fit an SVR model on the training set with linear kernel
svr_model_linear <- svm(SMOIS ~ TimeStamp, data = train_set, kernel="linear")

# Predict SMOIS values for the test set using linear kernel SVR model
svr_predictions_linear <- predict(svr_model_linear, ndata = test_set)

# Calculate the root mean squared error (RMSE) for the linear kernel SVR model
svr_rmse_linear <- sqrt(mean((test_set$SMOIS - svr_predictions_linear)^2))
cat("SVR_LINEAR RMSE:", svr_rmse_linear)


# Fit an SVR model on the training set with polynomial kernel
svr_model_poly <- svm(SMOIS ~ TimeStamp, data = train_set,kernel = "poly") 

# Predict TSK values for the test set using the polynomial kernel SVR model
svr_predictions_poly <- predict(svr_model_poly, ndata = test_set)

# Calculate the root mean squared error (RMSE) for the polynomial kernel SVR model
svr_rmse_poly <- sqrt(mean((test_set$SMOIS - svr_predictions_poly)^2))
cat("SVR_POLY RMSE:", svr_rmse_poly)



## RANDOM FOREST

# Load randomForest package
install.packages("randomForest")
library(randomForest)

# Fit a Random Forest model on the training set with ntree = 100
rf_model_100 <- randomForest(SMOIS ~ TimeStamp, data = train_set, ntree = 100)
# Display the Random Forest model summary
print(rf_model_100)
# Predict SMOIS values for the test set using the Random Forest model
rf_predictions_100 <- predict(rf_model_100, ndata = test_set)
# Calculate the root mean squared error (RMSE) for ntree = 100
rf_rmse_100 <- sqrt(mean((test_set$SMOIS - rf_predictions_100)^2))
cat("Random Forest RMSE for ntree=100:", rf_rmse_100, "\n")


# Fit a Random Forest model on the training set with ntree = 200
rf_model_200<- randomForest(SMOIS ~ TimeStamp, data = train_set, ntree = 200)
# Display the Random Forest model summary
# Predict SMOIS values for the test set using the Random Forest model
rf_predictions_200 <- predict(rf_model_200, ndata = test_set)
# Calculate the root mean squared error (RMSE) for ntree = 200
rf_rmse_200 <- sqrt(mean((test_set$SMOIS - rf_predictions_200)^2))
cat("Random Forest RMSE for ntree=200:", rf_rmse_200, "\n")


# Fit a Random Forest model on the training set with ntree = 500
rf_model_500<- randomForest(SMOIS ~ TimeStamp, data = train_set, ntree = 500)
# Display the Random Forest model summary
# Predict SMOIS values for the test set using the Random Forest model
rf_predictions_500 <- predict(rf_model_500, ndata = test_set)
# Calculate the root mean squared error (RMSE) for ntree = 500
rf_rmse_500 <- sqrt(mean((test_set$SMOIS - rf_predictions_500)^2))
cat("Random Forest RMSE for ntree=500:", rf_rmse_500, "\n")



# Create a dataframe with the RMSE values for ARIMA, Linear Regression, SVR Poly, and Random Forest (ntree=500)
rmse_comparison_df <- data.frame(
  Model = c("ARIMA", "Linear Regression", "SVR Poly", "Random Forest (ntree=500)"),
  RMSE = c(rmse_arima, LINEAR_REGRESSION_RMSE, svr_rmse_poly, rf_rmse_500)
)

# Bar chart to visualize the comparison
ggplot(rmse_comparison_df, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", width = 0.5) +
  labs(title = "Root Mean Squared Error for Selected Models",
       x = "Model",
       y = "Root Mean Squared Error") +
  theme_minimal() +
  scale_fill_manual(values = c("ARIMA" = "blue", "Linear Regression" = "green", "SVR Poly" = "red", "Random Forest (ntree=500)" = "purple")) +
  theme(legend.position = "none")






