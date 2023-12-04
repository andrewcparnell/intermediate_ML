# Load necessary libraries
library(keras)
library(tidyverse)

# Set seed for reproducibility
set.seed(123)

# Selecting the data
data <- mtcars[, c("mpg", "hp", "wt")]

# Splitting the data into training and test sets (70% training, 30% testing)
train_index <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Defining the response and covariates for training
x_train <- train_data %>% 
  select("hp", "wt") %>% 
  scale() %>% 
  as.matrix()
y_train <- train_data$mpg

# Defining the response and covariates for testing
x_test <- test_data %>% 
  select("hp", "wt") %>% 
  scale() %>% 
  as.matrix()
y_test <- test_data$mpg

# Building the neural network model
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = ncol(x_train)) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 1)

# Compiling the model
model %>% compile(
  optimizer = 'adam',
  loss = 'mse',
  metrics = c('mae')
)

# Training the model
model %>% fit(
  x_train,
  y_train,
  epochs = 200,
  batch_size = 16,
  validation_split = 0.2
)

# Predicting the test set
test_predictions <- model %>% predict(x_test)

# Plotting the test set predictions
plot(y_test, test_predictions, xlab="Actual MPG", ylab="Predicted MPG",
     main="Test Set Predictions")
abline(0, 1, col="red")

# Calculating the RMSE on the test set
rmse <- sqrt(mean((test_predictions - y_test)^2))
print(rmse)
