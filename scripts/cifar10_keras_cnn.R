# 1. Install and load necessary packages
# install.packages("keras")
library(keras)

# This data set contains 60,000 32 by 32 colour images
# 50k are training, 10k are test
# Labelled over 10 categories
# The classifications are aeroplane, automobile, etc

# 2. Load the CIFAR-10 dataset
cifar10 <- dataset_cifar10()
x_train <- cifar10$train$x
y_train <- cifar10$train$y
x_test <- cifar10$test$x
y_test <- cifar10$test$y

# Plot some of the training data
par(mfrow = c(2, 5), mar = c(0, 0, 1.5, 0), xaxs = "i", yaxs = "i")

# Looping through the first 10 images and plotting them
for (i in 1:10) {
  img <- x_train[i,,,] / 255
  img <- aperm(img, c(2, 1, 3)) # Re-arranging dimensions
  raster_matrix <- matrix(rgb(img[,,1], img[,,2], img[,,3]), nrow=32)
  raster_obj <- t(as.raster(raster_matrix))
  plot(0, type = "n", xlim = c(0, 1), ylim = c(0, 1), axes = FALSE, xlab = "", ylab = "")
  rasterImage(raster_obj, 0, 0, 1, 1)
}

# 3. Preprocess data
# Normalize pixel values
x_train <- x_train / 255
x_test <- x_test / 255

# One-hot encode the target variable
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# 4. Build the CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(32, 32, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# 5. Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = keras$optimizers$legacy$RMSprop(learning_rate = 0.0001),
  # optimizer = optimizer_rmsprop(learning_rate = 0.0001),
  metrics = c('accuracy')
)

# 6. Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 64,
  validation_split = 0.2
)

# 7. Evaluate the model's performance
score <- model %>% evaluate(x_test, y_test)
cat('Test loss:', score[[1]], '\n')
cat('Test accuracy:', score[[2]], '\n')

# 8. Look at the predictions
predictions <- model %>% predict(x_test) 
colnames(predictions) <- colnames(y_test) <- 1:10
predictions_final <- colnames(predictions)[max.col(predictions)]
y_test_final <- colnames(y_test)[max.col(y_test)]

confusion_matrix <- table(Predicted=predictions_final, Actual=y_test_final)
print(confusion_matrix)


