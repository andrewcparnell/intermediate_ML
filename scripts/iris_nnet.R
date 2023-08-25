# 1. Setting Up
# install.packages("nnet")
library(nnet)

# 2. Data Splitting
set.seed(123)  # for reproducibility
indices <- sample(1:nrow(iris), nrow(iris)*0.7)

train_data <- iris[indices,]
test_data <- iris[-indices,]

# 3. Model Fitting
model <- nnet(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
              data=train_data, size=5, rang=0.1, decay=5e-4, maxit=200)

# 4. Predictions & Performance Evaluation
predictions <- predict(model, test_data[,1:4], type="class")

# Creating the confusion matrix
confusion_matrix <- table(Predicted=predictions, Actual=test_data$Species)
print(confusion_matrix)

# Accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))
