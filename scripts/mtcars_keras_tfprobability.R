# Load necessary libraries
library(keras)
library(tfprobability)
# install_tfprobability()

# Set seed for reproducibility
set.seed(123)

# Selecting the data
data <- mtcars[, c("mpg", "hp", "wt")]

# Splitting the data into training and test sets (70% training, 30% testing)
train_index <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Normalizing the data
normalize <- function(x) {
  (x - mean(x)) / sd(x)
}

train_data[, -1] <- as.data.frame(lapply(train_data[, -1], normalize))
test_data[, -1] <- as.data.frame(lapply(test_data[, -1], normalize))

# Defining the response and covariates for training
x_train <- as.matrix(train_data[, c("hp", "wt")])
y_train <- train_data$mpg

# Defining the response and covariates for testing
x_test <- as.matrix(test_data[, c("hp", "wt")])
y_test <- test_data$mpg

bt <- reticulate::import("builtins")
RBFKernelFn <- reticulate::PyClass(
  "KernelFn",
  inherit = tensorflow::tf$keras$layers$Layer,
  list(
    `__init__` = function(self, ...) {
      kwargs <- list(...)
      super()$`__init__`(kwargs)
      dtype <- kwargs[["dtype"]]
      self$`_amplitude` = self$add_variable(initializer = initializer_zeros(),
                                            dtype = dtype,
                                            name = 'amplitude')
      self$`_length_scale` = self$add_variable(initializer = initializer_zeros(),
                                               dtype = dtype,
                                               name = 'length_scale')
      NULL
    },
    
    call = function(self, x, ...) {
      x
    },
    
    kernel = bt$property(
      reticulate::py_func(
        function(self)
          tfp$math$psd_kernels$ExponentiatedQuadratic(
            amplitude = tf$nn$softplus(array(0.1) * self$`_amplitude`),
            length_scale = tf$nn$softplus(array(2) * self$`_length_scale`)
          )
      )
    )
  )
)


# Building the probabilistic model
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = dim(x_train)[2]) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_variational_gaussian_process(1, 
                                     kernel_provide = RBFKernelFn())

# Compiling the model
model %>% compile(
  #optimizer = 'rmsprop',
  optimizer = keras$optimizers$legacy$RMSprop(learning_rate = 0.001),
  loss = 'mse',
  metrics = c('mae')
)

# Training the model
model %>% fit(
  x_train,
  y_train,
  epochs = 200,
  batch_size = 16,
  validation_split = 0.2
)

# Predicting the test set (mean and standard deviation)
test_predictions <- model %>% tfprobability::tfp$layers$default_mean_field_variational_gaussian_process_predictive_distribution(x_test)

# Extracting the mean and standard deviation
test_mean <- as.numeric(test_predictions$loc)
test_std <- sqrt(as.numeric(test_predictions$scale_diag))

# Plotting the test set predictions with uncertainty
plot(y_test, test_mean, xlab="Actual MPG", ylab="Predicted MPG",
     main="Test Set Predictions with Uncertainty")
arrows(y_test, test_mean - 1.96 * test_std,
       y_test, test_mean + 1.96 * test_std,
       angle=90, code=3, length=0.1, col="blue")

# Calculating the RMSE on the test set
rmse <- sqrt(mean((test_mean - y_test)^2))

# Printing the RMSE
cat("Root Mean Square Error (RMSE) on the test set:", rmse, "\n")
