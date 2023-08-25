# Load required libraries
library(keras)
library(datasets)

# Load the Air Passengers dataset
data("AirPassengers")
air_passengers <- as.numeric(AirPassengers)

# Normalize the data
min_val <- min(air_passengers)
max_val <- max(air_passengers)
air_passengers <- (air_passengers - min_val) / (max_val - min_val)

# Prepare the data for time series forecasting
time_steps <- 12
X <- NULL
y <- NULL
for (i in 1:(length(air_passengers) - time_steps)) {
  X <- rbind(X, air_passengers[i:(i + time_steps - 1)])
  y <- c(y, air_passengers[i + time_steps])
}

# Split into training and test sets
train_size <- floor(0.8 * nrow(X))
X_train <- X[1:train_size, , drop = FALSE]
y_train <- y[1:train_size]
X_test <- X[(train_size + 1):nrow(X), , drop = FALSE]
y_test <- y[(train_size + 1):length(y)]

# Reshape the data for the RNN
X_train <- array(X_train, dim = c(nrow(X_train), time_steps, 1))
X_test <- array(X_test, dim = c(nrow(X_test), time_steps, 1))

# Define the GRU model
model <- keras_model_sequential() %>%
  layer_gru(units = 50, activation = 'relu', input_shape = c(time_steps, 1)) %>%
  layer_dense(units = 1)

# Compile the model
model %>% compile(
  # optimizer = optimizer_adam(learning_rate = 0.001),
  optimizer = keras$optimizers$legacy$Adam(learning_rate = 0.001),
  loss = 'mse'
)

# Train the model
model %>% fit(
  x = X_train,
  y = y_train,
  epochs = 100,
  batch_size = 8,
  validation_split = 0.1
)

# Evaluate the model on test data
test_loss <- model %>% evaluate(X_test, y_test)
cat('Test Loss:', test_loss, '\n')

# Make predictions
predictions <- model %>% predict(X_test)

# Convert predictions back to original scale
predictions <- predictions * (max_val - min_val) + min_val

# You can further analyze or visualize the predictions as needed
y_test_gru <- y_test * (max_val - min_val) + min_val
plot(y_test_gru, predictions)
abline(a = 0, b = 1)

plot(y_test_gru)
lines(predictions, col = 'red')
