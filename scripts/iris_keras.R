# 1. Setting Up
# install.packages("keras")
library(keras)

# 2. Data Preparation
set.seed(123)  # for reproducibility
indices <- sample(1:nrow(iris), nrow(iris)*0.7)

train_data <- iris[indices,]
test_data <- iris[-indices,]

# One-hot encode the target variable
train_labels <- to_categorical(as.numeric(train_data$Species) - 1)
test_labels <- to_categorical(as.numeric(test_data$Species) - 1)

# 3. Model Building & Fitting
model <- keras_model_sequential() %>%
  layer_dense(units = 5, activation = 'relu', input_shape = 4) %>%
  layer_dense(units = 12, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')

model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  as.matrix(train_data[,1:4]), 
  train_labels,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.2
)
plot(history)

# 4. Predictions & Performance Evaluation
predictions <- model %>% predict(as.matrix(test_data[,1:4])) %>% k_argmax() %>% as.vector()
confusion_matrix <- table(Predicted=predictions, Actual=as.numeric(test_data$Species) - 1)
print(confusion_matrix)

# Accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))
