# Load the data
data(AirPassengers)
data <- as.numeric(AirPassengers)

# Normalize the data
max_val <- max(data)
min_val <- min(data)
data_normalized <- (data - min_val) / (max_val - min_val)

# Create sequences
look_back <- 3
dataX <- NULL
dataY <- NULL
for (i in 1:(length(data_normalized) - look_back)) {
  dataX <- rbind(dataX, data_normalized[i:(i+look_back-1)])
  dataY <- c(dataY, data_normalized[i + look_back])
}

# Split data
split_idx <- round(0.7 * nrow(dataX))
X_train <- dataX[1:split_idx, ]
y_train <- dataY[1:split_idx]
X_test <- dataX[(split_idx+1):nrow(dataX), ]
y_test <- dataY[(split_idx+1):length(dataY)]

# Initialize weights and biases
initialize_parameters <- function() {
  list(
    W_input_hidden = matrix(runif(look_back * 32, -1, 1), look_back),
    b_input_hidden = matrix(runif(32), nrow = 32),
    W_hidden_output = matrix(runif(32, -1, 1), 32),
    b_hidden_output = matrix(runif(1), 1)
  )
}

# Activation functions
sigmoid <- function(z) { 1 / (1 + exp(-z)) }
sigmoid_prime <- function(z) { sigmoid(z) * (1 - sigmoid(z)) }

# Forward and backward passes (simplified)
forward_backward_pass <- function(X, Y, params) {
  # Forward
  hidden = sigmoid(X %*% params$W_input_hidden + t(params$b_input_hidden))
  output = hidden %*% params$W_hidden_output + params$b_hidden_output
  
  # Calculate error (simplified for illustration)
  error = output - Y
  
  # Backward (gradients; simplified for illustration)
  d_output = error
  d_hidden = d_output %*% t(params$W_hidden_output) * sigmoid_prime(hidden)
  
  gradients = list(
    d_W_input_hidden = t(X) %*% d_hidden,
    d_b_input_hidden = colSums(d_hidden),
    d_W_hidden_output = t(hidden) %*% d_output,
    d_b_hidden_output = sum(d_output)
  )
  
  return(list(error=error, gradients=gradients))
}

# Training (simplified)
params <- initialize_parameters()
learning_rate <- 0.01

for(epoch in 1:100) {
  for(i in 1:nrow(X_train)) {
    result <- forward_backward_pass(matrix(X_train[i, ], ncol = look_back), y_train[i], params)
    
    # Update weights
    params$W_input_hidden = params$W_input_hidden - learning_rate * result$gradients$d_W_input_hidden
    params$b_input_hidden = params$b_input_hidden - learning_rate * result$gradients$d_b_input_hidden
    params$W_hidden_output = params$W_hidden_output - learning_rate * result$gradients$d_W_hidden_output
    params$b_hidden_output = params$b_hidden_output - learning_rate * result$gradients$d_b_hidden_output
  }
}

# Predictions (simplified)
predict <- function(X, params) {
  hidden = sigmoid(X %*% params$W_input_hidden + t(params$b_input_hidden))
  output = hidden %*% params$W_hidden_output + params$b_hidden_output
  return(output)
}

predictions_normalized <- predict(X_test, params)

# Convert predictions back to original scale
predictions <- predictions_normalized * (max_val - min_val) + min_val
