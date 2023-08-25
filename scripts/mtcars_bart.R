# Load the required library
library(BART)

# Set seed for reproducibility
set.seed(123)

# Selecting the data
data <- mtcars[, c("mpg", "hp", "wt")]

# Splitting the data into training and test sets (70% training, 30% testing)
train_index <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Defining the response and covariates for training
y.train <- train_data$mpg
x.train <- as.matrix(train_data[, c("hp", "wt")])

# Defining the covariates for testing
x.test <- as.matrix(test_data[, c("hp", "wt")])

# Fitting the BART model
bart_model <- wbart(x.train, y.train, x.test, nskip=100, ndpost=1000)

# Extracting the test set predictions
test_predictions <- bart_model$yhat.test.mean

# Plotting the test set predictions
plot(test_data$mpg, test_predictions, xlab="Actual MPG", ylab="Predicted MPG",
     main="Test Set Predictions")
abline(0, 1, col="red")

# Plotting the posterior uncertainty in the test set predictions
posterior_predictions <- t(bart_model$yhat.test)
posterior_mean <- rowMeans(posterior_predictions)
posterior_sd <- apply(posterior_predictions, 1, sd)

# Create 95% confidence intervals
posterior_low <- posterior_mean - 1.96 * posterior_sd
posterior_high <- posterior_mean + 1.96 * posterior_sd

# Now show the uncertainty
plot(test_data$mpg, posterior_mean, 
     ylim = range(c(posterior_low, posterior_high)),
     xlab="Actual MPG", ylab="Predicted MPG",
     main="Posterior Uncertainty in Test Set Predictions")
arrows(test_data$mpg, posterior_low,
       test_data$mpg, posterior_high,
       angle=90, code=3, length=0.1, col="blue")
abline(0, 1, col = 'red')

# RMSE
rmse <- sqrt(mean((test_predictions - test_data$mpg)^2))
print(rmse)
