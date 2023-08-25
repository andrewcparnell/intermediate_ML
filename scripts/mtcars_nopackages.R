# Load mtcars dataset
data(mtcars)

# Normalize function
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Splitting data into training and test sets
set.seed(123)
train_idx <- sample(1:nrow(mtcars), nrow(mtcars)*0.7)

train_data <- mtcars[train_idx, ]
test_data <- mtcars[-train_idx, ]

X_train <- as.matrix(train_data[, c("hp", "wt")])
y_train <- train_data$mpg
X_test <- as.matrix(test_data[, c("hp", "wt")])
y_test <- test_data$mpg

# Normalize data
X_train <- apply(X_train, 2, normalize)
y_train <- normalize(y_train)
X_test <- apply(X_test, 2, normalize)
y_test <- normalize(y_test)

# Initialize parameters
n_input <- 2
n_hidden <- 5
n_output <- 1

W1 <- matrix(runif(n_input * n_hidden, -1, 1), n_input)
b1 <- matrix(runif(n_hidden), n_hidden)
W2 <- matrix(runif(n_hidden * n_output, -1, 1), n_hidden)
b2 <- matrix(runif(n_output), n_output)

sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

sigmoid_prime <- function(z) {
  sigmoid(z) * (1 - sigmoid(z))
}

# Hyperparameters
learning_rate <- 0.01
epochs <- 10000

# Training the network on training data
for(epoch in 1:epochs) {
  z1 <- X_train %*% W1 + matrix(rep(b1, nrow(X_train)), ncol = n_hidden, byrow = TRUE)
  a1 <- sigmoid(z1)
  
  z2 <- a1 %*% W2 + matrix(rep(b2, nrow(X_train)), ncol = n_output, byrow = TRUE)
  a2 <- z2
  
  delta2 <- a2 - y_train
  delta1 <- (delta2 %*% t(W2)) * sigmoid_prime(z1)
  
  W2_gradient <- t(a1) %*% delta2
  b2_gradient <- colSums(delta2)
  
  W1_gradient <- t(X_train) %*% delta1
  b1_gradient <- colSums(delta1)
  
  W2 <- W2 - learning_rate * W2_gradient
  b2 <- b2 - learning_rate * b2_gradient
  
  W1 <- W1 - learning_rate * W1_gradient
  b1 <- b1 - learning_rate * b1_gradient
}

# Prediction function
predict_nn <- function(X) {
  z1 <- X %*% W1 + matrix(rep(b1, nrow(X)), ncol = n_hidden, byrow = TRUE)
  a1 <- sigmoid(z1)
  
  z2 <- a1 %*% W2 + matrix(rep(b2, nrow(X)), ncol = n_output, byrow = TRUE)
  return(z2)
}

# Predict on the test set
predictions <- predict_nn(X_test)

# Calculate RMSE
RMSE <- sqrt(mean((predictions - y_test)^2))

print(RMSE)

plot(y_test, predictions)
abline(a = 0, b = 1)
