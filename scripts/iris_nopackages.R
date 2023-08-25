# Data Preparation
set.seed(123)
indices <- sample(1:nrow(iris), nrow(iris)*0.7)

train_data <- as.matrix(iris[indices, -5])
test_data <- as.matrix(iris[-indices, -5])
train_labels <- iris[indices, 5]
test_labels <- iris[-indices, 5]

# Convert species to one-hot encoded vectors
to_one_hot <- function(labels) {
  model.matrix(~ labels - 1)  # The `-1` removes the intercept
}
train_labels_onehot <- to_one_hot(train_labels)
test_labels_onehot <- to_one_hot(test_labels)

# Neural Network Parameters
input_size <- ncol(train_data)
hidden_size <- 5
output_size <- 3
learning_rate <- 0.001
epochs <- 200

# Initialize weights and biases
W1 <- matrix(rnorm(input_size * hidden_size), input_size)
b1 <- matrix(0, hidden_size)

W2 <- matrix(rnorm(hidden_size * output_size), hidden_size)
b2 <- matrix(0, output_size)

# Activation and Softmax functions
relu <- function(x) {
  return(matrix(pmax(0, x), nrow=nrow(x), ncol=ncol(x)))
}
softmax <- function(x) exp(x) / rowSums(exp(x))

# Forward pass
forward <- function(x, W1, b1, W2, b2) {
  z1 <- x %*% W1 + matrix(rep(t(b1), nrow(x)), nrow(x), length(b1), byrow=TRUE)
  a1 <- relu(z1)
  z2 <- a1 %*% W2 + matrix(rep(t(b2), nrow(a1)), nrow(a1), length(b2), byrow=TRUE)
  a2 <- softmax(z2)
  return(list(a1=a1, a2=a2))
}

# Loss function: Categorical cross entropy
compute_loss <- function(y_pred, y_true) {
  -sum(y_true * log(y_pred)) / nrow(y_true)
}

# Backpropagation and weight update using gradient descent
for(epoch in 1:epochs) {
  # Forward pass
  activations <- forward(train_data, W1, b1, W2, b2)
  y_pred <- activations$a2
  
  # Compute loss
  loss <- compute_loss(y_pred, train_labels_onehot)
  if(epoch %% 50 == 0) cat('Epoch:', epoch, 'Loss:', loss, '\n')
  
  # Backpropagation (simplified for brevity)
  dz2 <- y_pred - train_labels_onehot
  dW2 <- t(activations$a1) %*% dz2
  db2 <- colSums(dz2)
  dz1 <- dz2 %*% t(W2) * (activations$a1 > 0)
  dW1 <- t(train_data) %*% dz1
  db1 <- colSums(dz1)
  
  # Update weights
  W1 <- W1 - learning_rate * dW1
  b1 <- b1 - learning_rate * db1
  W2 <- W2 - learning_rate * dW2
  b2 <- b2 - learning_rate * db2
}

# Predictions
test_activations <- forward(test_data, W1, b1, W2, b2)
predictions <- max.col(test_activations$a2) - 1
true_labels <- max.col(test_labels_onehot) - 1

# Performance Evaluation
confusion_matrix <- table(Predicted=predictions, Actual=true_labels)
print(confusion_matrix)

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, '\n')
