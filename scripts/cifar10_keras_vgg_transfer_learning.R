# Load the required libraries - set this running before talking about it
library(keras)

# Load the CIFAR-10 dataset
cifar10 <- dataset_cifar10()
x_train <- cifar10$train$x
y_train <- cifar10$train$y
x_test <- cifar10$test$x
y_test <- cifar10$test$y

# Preprocess the data
x_train <- x_train / 255
x_test <- x_test / 255

# Load the pre-trained VGG16 model without the top layers
base_model <- application_vgg16(weights = "imagenet",
                                include_top = FALSE,
                                input_shape = c(32, 32, 3))

# Freeze the base model layers
for (layer in base_model$layers) {
  layer$trainable <- FALSE
}

# Create a custom top layer for classification
top_model <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(512)) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")

# Combine the base and top models
model <- keras_model_sequential() %>%
  base_model %>%
  layer_flatten() %>%
  top_model

# Compile the model
model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = "accuracy"
)

# Train the model
history <- model %>% fit(
  x_train,
  y_train,
  epochs = 10,
  batch_size = 64,
  validation_split = 0.2
)

# Evaluate the model
evaluation <- model %>% evaluate(x_test, y_test)
cat("Test accuracy:", evaluation[2])

# Plot training history
plot(history)

# Look at cross tabulation
predictions <- model %>% predict(x_test) 
predictions_final <- predictions %>% k_argmax() %>% as.vector()

confusion_matrix <- table(Predicted = predictions_final, Actual = y_test)
print(confusion_matrix)

# Would be interesting at this point to look at which ones it got wrong
