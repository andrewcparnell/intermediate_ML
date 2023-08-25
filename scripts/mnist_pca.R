# Load Libraries
library(keras) # Just for the data set
library(stats)
library(ggplot2)

# Load Data
mnist <- dataset_mnist()
x_train <- mnist$train$x
x_test <- mnist$test$x

# Preprocess Data
x_train <- array_reshape(x_train / 255, c(nrow(x_train), 784))
x_test <- array_reshape(x_test / 255, c(nrow(x_test), 784))

# Plot the First Few Training Images
par(mfrow = c(2, 2))
for (i in 1:4) {
  img <- matrix(x_train[i,], nrow = 28)
  image(img[,ncol(img):1], axes = FALSE, main = paste("Label:", mnist$train$y[i]))
}

# Remove columns with zero variance
which_train_bad <- which(apply(x_train, 2, 'sd') == 0)
which_test_bad <- which(apply(x_test, 2, 'sd') == 0)
which_bad <- unique(c(which_train_bad, which_test_bad))
x_train <- x_train[, -which_bad]
x_test <- x_test[, -which_bad]

# Perform Principal Components Analysis
pca <- prcomp(x_train, center = TRUE, scale. = TRUE)

# Project Data onto First Two Principal Components
projected_x_test <- predict(pca, newdata = x_test)[,1:2]

# Create a Data Frame for Plotting
projected_df <- as.data.frame(projected_x_test)
projected_df$digit <- as.factor(mnist$test$y)

# Plot Encoded Space Using ggplot2
ggplot(projected_df, aes(x = PC1, y = PC2, color = digit)) +
  geom_point() +
  scale_color_discrete(name = "Digit") +
  labs(title = "2D Visualization of Encoded Space Using PCA",
       x = "Principal Component 1",
       y = "Principal Component 2")
