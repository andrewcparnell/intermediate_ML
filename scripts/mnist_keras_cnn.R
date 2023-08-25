# Install and load the keras package
# install.packages("keras")
library(keras)

# Load and preprocess the MNIST dataset
mnist <- dataset_mnist()

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Reshape and normalize the data
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
x_train <- x_train / 255
x_test <- x_test / 255

# One-hot encode the labels
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Build and compile the CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Train the model
history <- model %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 128, 
  validation_split = 0.2
)

# Evaluate the model
scores <- model %>% evaluate(x_test, y_test)
print(paste("Test loss:", scores[1]))
print(paste("Test accuracy:", scores[2]))

# Look at the predictions
predictions <- model %>% predict(x_test) 
colnames(predictions) <- colnames(y_test) <- 1:10
predictions_final <- colnames(predictions)[max.col(predictions)]
y_test_final <- colnames(y_test)[max.col(y_test)]

confusion_matrix <- table(Predicted=predictions_final, Actual=y_test_final)
print(confusion_matrix)

