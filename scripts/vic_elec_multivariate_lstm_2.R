# Load the required library
library(keras)
library(tidyverse)

set.seed(123)

# Note that this model uses Demand and Temperature as inputs to the 
# LSTM model but only predicts Demand

data("vic_elec", package = "tsibbledata")
data_matrix <- vic_elec %>% 
  select("Demand", "Temperature") %>% 
  # slice_head(n = 1000) %>% # Use this if you want to run faster
  as.matrix() %>%
  scale() # Remove this at your peril! (Or to learn why scaling is helpful)

# Data size
n_samples <- nrow(data_matrix)
n_features <- ncol(data_matrix)
n_output_features <- 1 # Only predicting Demand

# Define the sequence length and number of output features
sequence_length <- 10

# Create sequences for training
data_sequences <- list()
target_values <- list()  # Store target values

for (i in 1:(n_samples - sequence_length)) {
  data_sequences[[i]] <- data_matrix[i:(i + sequence_length - 1), ]
  target_values[[i]] <- data_matrix[i + sequence_length, 1]  # Predict the next values
}

data_sequences <- array_reshape(data_sequences, c(length(data_sequences), sequence_length, n_features))
target_values <- array_reshape(target_values, c(length(target_values), n_output_features))

# Create training and test data
train_ratio <- 0.8
train_idx <- floor(sample(1:nrow(target_values), 0.8 * nrow(target_values)))
train_sequences <- data_sequences[train_idx,,]
train_targets <- target_values[train_idx,]
test_sequences <- data_sequences[-train_idx,,]
test_targets <- target_values[-train_idx,]

# Define the LSTM model
model <- 
  keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(sequence_length, n_features)) %>%
  layer_dense(units = n_output_features)  # Output layer to predict demand

# Compile the model
model %>%
  compile(
    loss = 'mean_squared_error',
    optimizer = keras$optimizers$legacy$Adam(learning_rate = 0.01)
  )

# Train the model
history <- model %>%
  fit(
    x = train_sequences, 
    y = train_targets,
    epochs = 20,
    batch_size = 64,
    validation_data = list(test_sequences, test_targets)
  )
plot(history)

# Generate predictions
predictions <- model %>% predict(test_sequences)
head(predictions)

# Plot test set performance
plot(test_targets, predictions)
abline(a = 0, b = 1, col = 'red')
