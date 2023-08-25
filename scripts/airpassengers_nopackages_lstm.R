### Pseudocode for an LSTM:

# 1. **Initialize weights and biases** for input gate, forget gate, output gate, and cell state.
# 
# 2. **Forward pass**:
# 
#    - Compute the **forget gate**: \( f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \)
#    - Compute the **input gate**: \( i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \) and \( \tilde{C}_t = tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \)
#    - Update the **cell state**: \( C_t = f_t * C_{t-1} + i_t * \tilde{C}_t \)
#    - Compute the **output gate**: \( o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \)
#    - Update the **hidden state**: \( h_t = o_t * tanh(C_t) \)
# 
# 3. **Backward pass**:
#    - Compute gradients for each gate, state, and weights.
#    - Update weights using an optimization method.

data(AirPassengers)
data <- as.numeric(AirPassengers)
max_val <- max(data)
min_val <- min(data)
data_normalized <- (data - min_val) / (max_val - min_val)

sigmoid <- function(x) {
  1.0 / (1.0 + exp(-x))
}

tanh_activation <- function(x) {
  (exp(x) - exp(-x)) / (exp(x) + exp(-x))
}

input_dim <- 1
hidden_dim <- 1

# Initialize LSTM parameters
initialize_parameters <- function(input_dim, hidden_dim) {
  list(
    Wf = matrix(rnorm(input_dim + hidden_dim, hidden_dim), nrow = hidden_dim),
    Wi = matrix(rnorm(input_dim + hidden_dim, hidden_dim), nrow = hidden_dim),
    Wc = matrix(rnorm(input_dim + hidden_dim, hidden_dim), nrow = hidden_dim),
    Wo = matrix(rnorm(input_dim + hidden_dim, hidden_dim), nrow = hidden_dim),
    bf = matrix(0, hidden_dim, 1),
    bi = matrix(0, hidden_dim, 1),
    bc = matrix(0, hidden_dim, 1),
    bo = matrix(0, hidden_dim, 1)
  )
}

params <- initialize_parameters(input_dim, hidden_dim)

lstm_cell_forward <- function(xt, prev_h, prev_c, params) {
  concat_input <- rbind(prev_h, xt) # Note: Changing the order
  
  # Forget gate
  ft <- sigmoid(params$Wf %*% concat_input + params$bf)
  
  # Input gate
  it <- sigmoid(params$Wi %*% concat_input + params$bi)
  cct <- tanh_activation(params$Wc %*% concat_input + params$bc)
  
  # Update cell state
  ct <- ft * prev_c + it * cct
  
  # Output gate
  ot <- sigmoid(params$Wo %*% concat_input + params$bo)
  ht <- ot * tanh_activation(ct)
  
  list(ht = ht, ct = ct)
}

# Sample forward pass
prev_h <- matrix(0, hidden_dim, 1)
prev_c <- matrix(0, hidden_dim, 1)
xt <- matrix(data_normalized[1], input_dim, 1)

out <- lstm_cell_forward(xt, prev_h, prev_c, params)
print(out$ht)


