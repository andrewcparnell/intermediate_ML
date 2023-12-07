# Load necessary libraries
library(pdfCluster)
library(mclust)
library(ggplot2)

# Load the olive oil data
data(oliveoil)

# Transform the data first since it is proportions
# See help(oliveoil) and help(pdfCluster)
olive1 <- 1 + oliveoil[, 3:10]
margin <- apply(data.matrix(olive1),1,sum)
olive1 <- olive1/margin
alr <- (-log( olive1[, -4]/olive1[, 4]))

# Create some basic visualizations
pairs(olive1, main = "Scatterplot Matrix of Olive Oil Data")

# Run model-based clustering with different covariance types
covariance_types <- c("EII", "VII", "EEI", "VEI", "EVI", "VVI")
models <- lapply(covariance_types, function(cov_type) {
  Mclust(olive1, G = 1:5, modelNames = cov_type)
})

# Model selection based on BIC
BIC_values <- sapply(models, function(model) model$BIC)
which_best_model <- which(BIC_values == max(BIC_values),
                          arr.ind = TRUE)
best_model <- Mclust(olive1, G = which_best_model[2], 
                     modelNames = covariance_types[which_best_model[1]])

# Print the best model
summary(best_model)

# Plot the output of the best model
plot(best_model, what = 'classification')

# Compare BIC
all_mods <- Mclust(olive1, G = 1:10)
plot(all_mods, what = 'BIC')

# Extract out the probabilities of each observation being in each cluster
head(best_model$z)
