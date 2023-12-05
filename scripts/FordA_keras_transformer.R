# Load necessary libraries
library(tensorflow)
library(keras)

# Set seed for reproducibility
set.seed(1234)

# Define the URL for the dataset
url <- "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA"

# Load the training data
train_df <- "FordA_TRAIN.tsv" %>%
  get_file(., file.path(url, .)) %>%
  readr::read_tsv(col_names = FALSE)

# Extract features and labels from the training data
x_train <- as.matrix(train_df[, -1])
y_train <- as.matrix(train_df[, 1])

# Load the test data
test_df <- "FordA_TEST.tsv" %>%
  get_file(., file.path(url, .)) %>%
  readr::read_tsv(col_names = FALSE)

# Extract features and labels from the test data
x_test <- as.matrix(test_df[, -1])
y_test <- as.matrix(test_df[, 1])

# Determine the number of unique classes in the training data
n_classes <- length(unique(y_train))

# Shuffle the training data
shuffle_ind <- sample(nrow(x_train))
x_train <- x_train[shuffle_ind, , drop = FALSE]
y_train <- y_train[shuffle_ind, , drop = FALSE]

# Replace -1 labels with 0 in both training and test data
y_train[y_train == -1] <- 0
y_test [y_test  == -1] <- 0

# Add a third dimension to the input data
dim(x_train) <- c(dim(x_train), 1)
dim(x_test) <- c(dim(x_test), 1)

# Define a function for the Transformer encoder
transformer_encoder <- function(inputs,
                                head_size,
                                num_heads,
                                ff_dim,
                                dropout = 0) {
  # Attention and Normalization
  attention_layer <-
    layer_multi_head_attention(key_dim = head_size,
                               num_heads = num_heads,
                               dropout = dropout)
  
  n_features <- dim(inputs) %>% tail(1)
  
  x <- inputs %>%
    attention_layer(., .) %>%
    layer_dropout(dropout) %>%
    layer_layer_normalization(epsilon = 1e-6)
  
  res <- x + inputs
  
  # Feed Forward Part
  x <- res %>%
    layer_conv_1d(ff_dim, kernel_size = 1, activation = "relu") %>%
    layer_dropout(dropout) %>%
    layer_conv_1d(n_features, kernel_size = 1) %>%
    layer_layer_normalization(epsilon = 1e-6)
  
  # return output + res
  x + res
}

# Define a function to build the model
build_model <- function(input_shape,
                        head_size,
                        num_heads,
                        ff_dim,
                        num_transformer_blocks,
                        mlp_units,
                        dropout = 0,
                        mlp_dropout = 0) {
  
  inputs <- layer_input(input_shape)
  
  x <- inputs
  for (i in 1:num_transformer_blocks) {
    x <- x %>%
      transformer_encoder(
        head_size = head_size,
        num_heads = num_heads,
        ff_dim = ff_dim,
        dropout = dropout
      )
  }
  
  x <- x %>% 
    layer_global_average_pooling_1d(data_format = "channels_first")
  
  for (dim in mlp_units) {
    x <- x %>%
      layer_dense(dim, activation = "relu") %>%
      layer_dropout(mlp_dropout)
  }
  
  outputs <- x %>% 
    layer_dense(n_classes, activation = "softmax")
  
  keras_model(inputs, outputs)
}

# Get the input shape from the training data
input_shape <- dim(x_train)[-1] # drop batch dim

# Build the model
model <- build_model(
  input_shape,
  head_size = 256,
  num_heads = 4,
  ff_dim = 4,
  num_transformer_blocks = 4,
  mlp_units = c(128),
  mlp_dropout = 0.4,
  dropout = 0.25
)

# Compile the model
model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 1e-4),
  metrics = c("sparse_categorical_accuracy")
)

# Display model summary
model

# Define early stopping callbacks
callbacks <- list(
  callback_early_stopping(patience = 10, restore_best_weights = TRUE)
)

# Train the model
history <- model %>%
  fit(
    x_train,
    y_train,
    batch_size = 64,
    epochs = 10, # Originally 200
    callbacks = callbacks,
    validation_split = 0.2
  )

# Evaluate the model on the test data
model %>% evaluate(x_test, y_test, verbose = 1)
