# Load required libraries
library(keras)
library(datasets)

# Load the dataset
data("EuStockMarkets")

# Using DAX index
stock_data <- EuStockMarkets[, "DAX"]

# Create binary labels (up or down)
labels <- ifelse(diff(stock_data) > 0, 1, 0)

# Create sequences of data (e.g., window of 5 days)
sequence_length <- 5
data_sequences <- list()
for (i in 1:(length(stock_data) - sequence_length - 1)) {
  data_sequences[[i]] <- stock_data[i:(i + sequence_length - 1)]
}

data_sequences <- do.call(rbind, data_sequences)
labels <- labels[sequence_length:length(labels) - 1]

# Split data into training and testing sets
train_size <- floor(0.8 * nrow(data_sequences))
x_train <- data_sequences[1:train_size,]
y_train <- labels[1:train_size]
x_test <- data_sequences[(train_size + 1):nrow(data_sequences),]
y_test <- labels[(train_size + 1):nrow(data_sequences)]

# Reshape data for GRU
x_train <- array_reshape(x_train, c(nrow(x_train), sequence_length, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), sequence_length, 1))

# Define model
model <- keras_model_sequential() %>%
  layer_gru(units = 64, input_shape = c(sequence_length, 1)) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile model
model %>% compile(
  optimizer = keras$optimizers$legacy$Adam(learning_rate = 0.001),
  # optimizer = optimizer_adam(lr = 0.001),
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

# Train model
history <- model %>% fit(
  x_train, y_train,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate model
score <- model %>% evaluate(x_test, y_test)
cat('Test loss:', score[[1]], '\n')
cat('Test accuracy:', score[[2]], '\n')
