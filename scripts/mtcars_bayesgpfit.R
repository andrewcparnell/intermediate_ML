# Load necessary libraries
library(BayesGPfit)

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

# Fitting the Bayesian Gaussian Process model
gp_model <- GP.Bayes.fit(matrix(y_train, ncol = 1), x_train, 
                         poly_degree = 5L, 
                         a = 0.001,
                         b = 2,
                         progress_bar = TRUE)

# Predicting the test set
test_predictions <- GP.predict(gp_model,
                               x_test,
                               CI = TRUE)

# Extracting the mean of the predictions
posterior_mean <- test_predictions$mean$f

# Plotting the test set predictions
plot(y_test, posterior_mean, xlab="Actual MPG", ylab="Predicted MPG",
     main="Test Set Predictions")
abline(0, 1, col = 'red')

# Calculating the RMSE on the test set
rmse <- sqrt(mean((posterior_mean - y_test)^2))
print(rmse)

# Create 95% confidence intervals
posterior_low <- test_predictions$lci$f
posterior_high <- test_predictions$uci$f

# Now show the uncertainty
plot(test_data$mpg, posterior_mean, 
     ylim = range(c(posterior_low, posterior_high)),
     xlab="Actual MPG", ylab="Predicted MPG",
     main="Posterior Uncertainty in Test Set Predictions")
arrows(test_data$mpg, posterior_low,
       test_data$mpg, posterior_high,
       angle=90, code=3, length=0.1, col="blue")
abline(0, 1, col = 'red')
