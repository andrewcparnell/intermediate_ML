# Load required libraries
library(keras)
library(datasets)

# Load the Earthquake dataset
data("quakes")

# Extracting the magnitude as a feature
quakes_data <- quakes$mag

# Create binary labels (above or below mean magnitude)
mean_mag <- mean(quakes_data)
labels <- ifelse(quakes_data > mean_mag, 1, 0)

# Create sequences of data (e.g., window of 5 observations)
sequence_length <- 5
data_sequences <- list()
for (i in 1:(length(quakes_data) - sequence_length)) {
  data_sequences[[i]] <- quakes_data[i:(i + sequence_length - 1)]
}

data_sequences <- do.call(rbind, data_sequences)
labels <- labels[(sequence_length + 1):length(labels)]

# Split data into training and testing sets
train_size <- floor(0.8 * nrow(data_sequences))
x_train <- array_reshape(data_sequences[1:train_size,], c(train_size, sequence_length, 1))
y_train <- labels[1:train_size]
x_test <- array_reshape(data_sequences[(train_size + 1):nrow(data_sequences),], c(nrow(data_sequences) - train_size, sequence_length, 1))
y_test <- labels[(train_size + 1):length(labels)]

# Define LSTM model
model <- keras_model_sequential() %>%
  layer_lstm(units = 64, input_shape = c(sequence_length, 1)) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile model
model %>% compile(
  #optimizer = optimizer_adam(lr = 0.001),
  optimizer = keras$optimizers$legacy$Adam(learning_rate = 0.001),
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

# Train model
history <- model %>% fit(
  x_train, y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate model
score <- model %>% evaluate(x_test, y_test)
cat('Test loss:', score[[1]], '\n')
cat('Test accuracy:', score[[2]], '\n')
