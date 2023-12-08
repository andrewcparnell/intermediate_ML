# Write a script that takes in a static image and a set of features over time
# And outputs a multivariate set of values (potentially also an image)

# Load in packages
library(tidyverse)
library(keras)

# Make up some data - note that this will not fit well due to it being totally random
N_samples <- 500 # Total number of e.g. days
N_input_features <- 5 # E.g. number of weather variables
N_output_features <- 20 # number of output pixels
image_dim <- 32 # Dimension of static image

# Create the static image
static_image <- matrix(runif(image_dim^2), 
                       ncol = image_dim, 
                       nrow = image_dim)

# Create the feature set
X <- matrix(rnorm(N_samples * N_input_features), 
            ncol = N_input_features, 
            nrow = N_samples)

# Create the outputs
y <- matrix(rnorm(N_samples * N_output_features),
            ncol = N_output_features,
            nrow = N_samples)

# Define the sequence length
sequence_length <- 10

# Create sequences for training
data_sequences <- list()
target_values <- list()  # Store target values

for (i in 1:(N_samples - sequence_length)) {
  data_sequences[[i]] <- X[i:(i + sequence_length - 1), ]
  target_values[[i]] <- y[i + sequence_length, ]  # Predict the next values
}

# Re-shape the image
static_image_all <- list()
for (i in 1:(N_samples - sequence_length)) {
  static_image_all[[i]] <- static_image
}
static_img_arr <- array_reshape(static_image_all, 
                                c(length(data_sequences), 
                                  image_dim, image_dim, 1))

data_sequences <- array_reshape(data_sequences, 
                                c(length(data_sequences), 
                                  sequence_length, N_input_features))
target_values <- array_reshape(target_values, 
                               c(length(target_values), 
                                 N_output_features))

# Create training and test data
train_ratio <- 0.8
train_idx <- floor(sample(1:nrow(target_values), 0.8 * nrow(target_values)))
train_sequences <- data_sequences[train_idx,,]
train_targets <- target_values[train_idx,]
train_img <- static_img_arr[train_idx,,,]
test_sequences <- data_sequences[-train_idx,,]
test_targets <- target_values[-train_idx,]
test_img <- static_img_arr[-train_idx,,,]

# Define a model for the time series piece
model_lstm_input <- layer_input(shape = c(sequence_length, N_input_features))
model_img_input <- layer_input(shape = c(image_dim, image_dim, 1))

model_lstm <- model_lstm_input %>% 
  layer_lstm(units = 64) %>%
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.5)

model_img <- model_img_input %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten()

out <- layer_concatenate(list(model_lstm, model_img)) %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = N_output_features)

model <- keras_model(inputs = list(model_lstm_input, model_img_input),
                     outputs = out)

# Compile the model
model %>%
  compile(
    loss = 'mean_squared_error',
    optimizer = 'adam'
  )

# Train the model
history <- model %>%
  fit(
    x = list(train_sequences, train_img),
    y = train_targets,
    epochs = 20,
    batch_size = 64,
    validation_data = list(list(test_sequences, test_img), test_targets)
  )
plot(history)

# Get predictions
predictions <- model %>% predict(list(test_sequences, test_img))
