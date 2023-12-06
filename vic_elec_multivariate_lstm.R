# Load the required libraries
library(tsibble)
library(keras)
library(tidyverse)

# Load the vic_elec dataset
data("vic_elec", package = "tsibbledata")

# Preprocess the data
vic_elec <- vic_elec %>%
  as_tsibble() %>%
  fill_gaps(value = list(demand = 0, temperature = 0)) %>%
  stretch_tsibble(.init = 7, .step = 1)

# Normalize the data
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

vic_elec <- vic_elec %>%
  mutate_at(vars(demand, temperature), normalize)

# Split the data into training and testing sets
train_ratio <- 0.8
train_size <- floor(train_ratio * nrow(vic_elec))

train_data <- vic_elec[1:train_size, ]
test_data <- vic_elec[(train_size + 1):nrow(vic_elec), ]

# Create sequences for training and testing
sequence_length <- 10

train_sequences <- array_reshape(
  lapply(1:(nrow(train_data) - sequence_length + 1), function(i) {
    train_data[i:(i + sequence_length - 1), c("demand", "temperature")]
  }),
  c(sequence_length, 2)
)

test_sequences <- array_reshape(
  lapply(1:(nrow(test_data) - sequence_length + 1), function(i) {
    test_data[i:(i + sequence_length - 1), c("demand", "temperature")]
  }),
  c(sequence_length, 2)
)

# Define the LSTM model
model <- keras_model_sequential()

model %>%
  layer_lstm(units = 50, input_shape = c(sequence_length, 2)) %>%
  layer_dense(units = 2)

# Compile the model
model %>%
  compile(
    loss = "mean_squared_error",
    optimizer = optimizer_adam(),
    metrics = list("mean_absolute_error")
  )

# Train the model
history <- model %>%
  fit(
    train_sequences,
    train_sequences,
    epochs = 50,
    batch_size = 32,
    validation_split = 0.2
  )

# Evaluate the model
eval_result <- model %>% evaluate(test_sequences, test_sequences)

# Print evaluation results
cat("Test Loss:", eval_result$loss, "\n")
cat("Test MAE:", eval_result$mean_absolute_error, "\n")

# Predict using the trained model
predictions <- model %>% predict(test_sequences)

# Restore the original scale of the data if needed
# denormalize <- function(x) {
#   x * (max(train_data$demand) - min(train_data$demand)) + min(train_data$demand)
# }

# Plot actual vs. predicted values
plot(test_data$temperature[(sequence_length + 1):nrow(test_data)], type = "l", col = "blue", xlab = "Time", ylab = "Temperature")
lines(test_data$demand[(sequence_length + 1):nrow(test_data)], col = "red")
lines(predictions[, 2], col = "blue", lty = 2)
lines(predictions[, 1], col = "red", lty = 2)
legend("topright", legend = c("Actual Temperature", "Actual Demand", "Predicted Temperature", "Predicted Demand"), col = c("blue", "red", "blue", "red"), lty = c(1, 1, 2, 2))
