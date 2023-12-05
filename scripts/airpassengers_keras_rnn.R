library(keras)
library(datasets)

# Load the data
data(AirPassengers)
data <- as.numeric(AirPassengers)

# Normalize the data
max_val <- max(data)
min_val <- min(data)
data_normalized <- (data - min_val) / (max_val - min_val)

# Create sequences of length 'look_back' to predict the next value
look_back <- 3
dataX <- NULL
dataY <- NULL
for (i in 1:(length(data_normalized) - look_back)) {
  dataX <- rbind(dataX, data_normalized[i:(i+look_back-1)])
  dataY <- c(dataY, data_normalized[i + look_back])
}

# Splitting into train and test
split_idx <- round(0.7 * nrow(dataX))
X_train <- array(dataX[1:split_idx, ], dim = c(split_idx, look_back, 1))
y_train <- dataY[1:split_idx]
X_test <- array(dataX[(split_idx+1):nrow(dataX), ], dim = c(nrow(dataX) - split_idx, look_back, 1))
y_test <- dataY[(split_idx+1):length(dataY)]

# Build RNN model
model <- keras_model_sequential() %>%
  layer_simple_rnn(units = 32, input_shape = list(look_back, 1)) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(learning_rate = 0.01) # keras$optimizers$legacy$Adam(learning_rate = 0.001),
  # optimizer = optimizer_adam(learning_rate = 0.01)
)

# Train the model
history <- model %>% fit(
  X_train, y_train,
  epochs = 100,
  batch_size = 16,
  validation_data = list(X_test, y_test)
)

# Predict on the test set
predictions_normalized <- model %>% predict(X_test)

# Convert predictions back to original scale
predictions <- predictions_normalized * (max_val - min_val) + min_val

# You can now compare 'predictions' with the actual test values in 'AirPassengers'
y_test_nn <- y_test * (max_val - min_val) + min_val
plot(y_test_nn, predictions)
abline(a = 0, b = 1)
