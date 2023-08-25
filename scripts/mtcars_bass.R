# Load necessary libraries
library(BASS)

# Set seed for reproducibility
set.seed(123)

# Selecting the data
data <- mtcars[, c("mpg", "hp", "wt")]

# Splitting the data into training and test sets (70% training, 30% testing)
train_index <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Defining the response and covariates for training
x_train <- as.matrix(train_data[, c("hp", "wt")])
y_train <- train_data$mpg

# Defining the covariates for testing
x_test <- as.matrix(test_data[, c("hp", "wt")])
y_test <- test_data$mpg

# Fitting the Bayesian Adaptive Smoothing Spline model
bass_model <- bass(x_train, y_train)

# Predicting the test set
test_predictions <- predict(bass_model, x_test)

# Extracting the mean of the predictions
posterior_mean <- colMeans(test_predictions)

# Plotting the test set predictions
plot(y_test, posterior_mean, xlab="Actual MPG", ylab="Predicted MPG",
     main="Test Set Predictions")
abline(a = 0, b = 1, col = 'red')

# Calculating the RMSE on the test set
rmse <- sqrt(mean((test_mean - y_test)^2))
print(rmse)

# Create 95% confidence intervals
posterior_low <- apply(test_predictions, 2, 'quantile', 0.025)
posterior_high <- apply(test_predictions, 2, 'quantile', 0.975)

# Now show the uncertainty
plot(test_data$mpg, posterior_mean, 
     ylim = range(c(posterior_low, posterior_high)),
     xlab="Actual MPG", ylab="Predicted MPG",
     main="Posterior Uncertainty in Test Set Predictions")
arrows(test_data$mpg, posterior_low,
       test_data$mpg, posterior_high,
       angle=90, code=3, length=0.1, col="blue")
abline(0, 1, col = 'red')
