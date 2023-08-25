# 1. Setting Up
# install.packages("tensorflow")
library(tensorflow)

# 2. Data Preparation
set.seed(123)  # for reproducibility
indices <- sample(1:nrow(iris), nrow(iris)*0.7)

train_data <- iris[indices,]
test_data <- iris[-indices,]

# One-hot encode the target variable
train_labels <- tf$one_hot(tf$cast(as.numeric(train_data$Species) - 1, 
                                   dtype = "int32"), 
                           depth = 3L)
test_labels <- tf$one_hot(tf$cast(as.numeric(test_data$Species) - 1, 
          dtype = "int32"), depth = 3L)

# 3. Model Building & Fitting
W1 <- tf$Variable(tf$zeros(shape(4L, 5L)))
b1 <- tf$Variable(tf$zeros(shape(5L)))

W2 <- tf$Variable(tf$zeros(shape(5L, 3L)))
b2 <- tf$Variable(tf$zeros(shape(3L)))

x <- tf$placeholder(tf$float32, shape(NULL, 4L))
y <- tf$placeholder(tf$float32, shape(NULL, 3L))

layer1 <- tf$nn$relu(tf$matmul(x, W1) + b1)
logits <- tf$matmul(layer1, W2) + b2
predicted <- tf$argmax(logits, 1L)

loss <- tf$reduce_mean(tf$nn$softmax_cross_entropy_with_logits(logits = logits, labels = y))
optimizer <- tf$train$RMSPropOptimizer(0.001)$minimize(loss)

sess <- tf$Session()
sess$run(tf$global_variables_initializer())

# Training
for (i in 1:200) {
  sess$run(optimizer, feed_dict = dict(x = as.matrix(train_data[,1:4]), y = train_labels))
}

# 4. Predictions & Performance Evaluation
predictions <- sess$run(predicted, feed_dict = dict(x = as.matrix(test_data[,1:4])))

confusion_matrix <- table(Predicted=predictions, Actual=as.numeric(test_data$Species) - 1)
print(confusion_matrix)

# Accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))
