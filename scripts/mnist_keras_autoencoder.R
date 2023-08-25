# Load Libraries
library(keras)
library(ggplot2)
set.seed(123)

# Load Data
mnist <- dataset_mnist()
x_train <- mnist$train$x
x_test <- mnist$test$x

# Preprocess Data
x_train <- array_reshape(x_train / 255, c(nrow(x_train), 784))
x_test <- array_reshape(x_test / 255, c(nrow(x_test), 784))

# Define Autoencoder Model
encoding_dim <- 32
input_img <- layer_input(shape = c(784))

encoded <- input_img %>%
  layer_dense(units = encoding_dim, activation = 'relu')

decoded <- encoded %>%
  layer_dense(units = 784, activation = 'sigmoid')

autoencoder <- keras_model(inputs = input_img, outputs = decoded)

# Compile and Fit the Model
autoencoder %>% compile(
  optimizer = keras$optimizers$legacy$Adam(learning_rate = 0.01),
  #optimizer = 'adam',
  loss = 'binary_crossentropy'
)

autoencoder %>% fit(
  x_train, x_train,
  epochs = 50,
  batch_size = 256,
  validation_data = list(x_test, x_test)
)

# Create Encoder Model
encoder <- keras_model(inputs = autoencoder$input, outputs = encoded)
encoded_imgs <- predict(encoder, x_test)

# Plot Encoded Space
col_sds <- apply(encoded_imgs, 2, 'sd')
o <- order(col_sds, decreasing = TRUE)
encoded_df <- as.data.frame(encoded_imgs[,o[1:2]])
encoded_df$digit <- mnist$test$y

ggplot(encoded_df, aes(x = V1, y = V2, color = as.factor(digit))) +
  geom_point() +
  scale_color_discrete(name = "Digit") +
  labs(title = "2D Visualization of Encoded Space",
       x = paste("Dimension",o[1]),
       y = paste("Dimension",o[2]))
