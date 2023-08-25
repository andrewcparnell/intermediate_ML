# CURRENTLY NOT WORKING

# Load Libraries
library(keras)

# Load Data
mnist <- dataset_mnist()
x_train <- mnist$train$x
x_test <- mnist$test$x

# Preprocess Data
x_train <- array_reshape(x_train / 255, c(nrow(x_train), 784))
x_test <- array_reshape(x_test / 255, c(nrow(x_test), 784))

# Define Parameters
original_dim <- 784
intermediate_dim <- 256
latent_dim <- 2

# Encoder Network
inputs <- layer_input(shape = c(original_dim))
h <- layer_dense(inputs, intermediate_dim, activation = "relu")
z_mean <- layer_dense(h, latent_dim)
z_log_var <- layer_dense(h, latent_dim)

# Sampling Function
z <- layer_lambda(list(z_mean, z_log_var),
                  function(args) {
                    z_mean <- args[[1]]
                    z_log_var <- args[[2]]
                    epsilon <- k_random_normal(shape = k_shape(z_mean))
                    z_mean + k_exp(z_log_var / 2) * epsilon
                  })

# Encoder Model
encoder <- keras_model(inputs, list(z_mean, z_log_var, z))

# Decoder Network
decoder_input <- layer_input(shape = c(latent_dim))
decoder_h <- layer_dense(decoder_input, intermediate_dim, activation = "relu")
decoder_mean <- layer_dense(decoder_h, original_dim, activation = "sigmoid")
decoder <- keras_model(decoder_input, decoder_mean)

# VAE Model
vae_output <- decoder(z)
vae <- keras_model(inputs, vae_output)

# Loss Function
reconstruction_loss <- original_dim * loss_binary_crossentropy(inputs, vae_output)
kl_loss <- -0.5 * k_sum(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1)

custom_vae_loss <- function(inputs, vae_output) {
  reconstruction_loss <- original_dim * loss_binary_crossentropy(inputs, vae_output)
  kl_loss <- -0.5 * k_sum(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1)
  k_mean(reconstruction_loss + kl_loss)
}

# Compile and Fit the Model
vae %>% compile(optimizer = 'adam', loss = custom_vae_loss)
vae %>% fit(x_train, x_train, epochs = 50, batch_size = 256, validation_data = list(x_test, x_test))

# Compile and Fit the Model
vae %>% compile(optimizer = 'adam', loss = custom_vae_loss)
vae %>% fit(x_train, x_train, epochs = 50, batch_size = 256, validation_data = list(x_test, x_test))
# Generate New Data
n <- 15
digit_size <- 28
grid_x <- seq(qnorm(0.05), qnorm(0.95), length.out = n)
grid_y <- seq(qnorm(0.05), qnorm(0.95), length.out = n)

# Empty Grid for Displaying Images
grid <- matrix(0, n * digit_size, n * digit_size)

for (i in 1:n) {
  for (j in 1:n) {
    z_sample <- matrix(c(grid_x[i], grid_y[j]), nrow = 1)
    x_decoded <- as.array(predict(decoder, z_sample))
    digit <- array_reshape(x_decoded, c(digit_size, digit_size))
    grid[(i-1)*digit_size + 1:i*digit_size, (j-1)*digit_size + 1:j*digit_size] <- digit
  }
}

# Plot Generated Images
image(grid, axes = FALSE)
