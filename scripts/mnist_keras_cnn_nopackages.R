# Load MNIST dataset
data(mnist)

# Simplified data preprocessing
x_train <- mnist$train$x / 255
x_train <- matrix(x_train, ncol = 28*28)
y_train <- mnist$train$y

x_test <- mnist$test$x / 255
x_test <- matrix(x_test, ncol = 28*28)
y_test <- mnist$test$y

# One-hot encode function
one_hot <- function(y) {
  matrix(0, nrow=length(y), ncol=10)[cbind(1:length(y), y + 1)] <- 1
  matrix(0, nrow=length(y), ncol=10) + 1
}

y_train <- one_hot(y_train)
y_test <- one_hot(y_test)

# Convolution and Activation Functions
conv2d <- function(image, filter) {
  image_height <- dim(image)[1]
  image_width <- dim(image)[2]
  
  output_height <- image_height - dim(filter)[1] + 1
  output_width <- image_width - dim(filter)[2] + 1
  
  output <- matrix(0, nrow = output_height, ncol = output_width)
  
  for (i in 1:output_height) {
    for (j in 1:output_width) {
      region <- image[i:(i + dim(filter)[1] - 1), j:(j + dim(filter)[2] - 1)]
      output[i, j] <- sum(region * filter)
    }
  }
  return(output)
}

relu <- function(x) {
  return(pmax(0, x))
}

sigmoid <- function(z) {
  1.0 / (1.0 + exp(-z))
}

sigmoid_prime <- function(z) {
  sigmoid(z) * (1 - sigmoid(z))
}

# Network Setup
conv_filter <- matrix(rnorm(3*3), ncol = 3)
input_size <- 26 * 26  # Size after 3x3 convolution on a 28x28 image
hidden_size <- 128
output_size <- 10
learning_rate <- 0.01

W1 <- matrix(rnorm(input_size * hidden_size), ncol = hidden_size)
b1 <- matrix(0, ncol = hidden_size)
W2 <- matrix(rnorm(hidden_size * output_size), ncol = output_size)
b2 <- matrix(0, ncol = output_size)

# Training loop
for (epoch in 1:10) {
  for (i in 1:nrow(x_train)) {
    # Convert the flat image into 28x28 again
    image <- matrix(x_train[i,], nrow = 28, byrow = TRUE)
    
    # Convolutional forward pass
    conv_output <- conv2d(image, conv_filter)
    conv_output <- relu(conv_output)  # Apply ReLU activation
    
    # Flatten the conv_output to feed to fully connected layer
    input_to_fc <- as.vector(t(conv_output))
    
    # Forward pass for fully connected layers
    z1 <- input_to_fc %*% W1 + b1
    a1 <- sigmoid(z1)
    z2 <- a1 %*% W2 + b2
    a2 <- sigmoid(z2)
    
    # Compute loss and gradients using backpropagation
    loss <- sum((a2 - y_train[i,])^2) / 2
    delta2 <- (a2 - y_train[i,]) * sigmoid_prime(z2)
    delta1 <- delta2 %*% t(W2) * sigmoid_prime(z1)
    
    W2_gradient <- t(a1) %*% delta2
    b2_gradient <- delta2
    W1_gradient <- t(input_to_fc) %*% delta1
    b1_gradient <- delta1
    
    # Update weights and biases
    W2 <- W2 - learning_rate * W2_gradient
    b2 <- b2 - learning_rate * b2_gradient
    W1 <- W1 - learning_rate * W1_gradient
    b1 <- b1 - learning_rate * b1_gradient
  }
  print(paste("Epoch", epoch, "Loss:", loss))
}

# Evaluate
correct_predictions <- 0
for (i in 1:nrow(x_test)) {
  image <- matrix(x_test[i,], nrow = 28, byrow = TRUE)
  conv_output <- relu(conv2d(image, conv_filter))
  input_to_fc <- as.vector(t(conv_output))
  
  z1 <- input_to_fc %*% W1 + b1
  a1 <- sigmoid(z1)
  z2 <- a1 %*% W2 + b2
  a2 <- sigmoid(z2)
  
  if (which.max(a2) == which.max(y_test[i,])) {
    correct_predictions <- correct_predictions + 1
  }
}

accuracy <- correct_predictions / nrow(x_test)
print(paste("Accuracy:", accuracy))
