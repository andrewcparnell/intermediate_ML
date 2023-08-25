# Load necessary libraries
library(pdfCluster)
library(Rtsne)
library(fpc) # For cluster.stats

# Load the olive oil data
data(oliveoil)

# Transform the data first since it is proportions
# See help(oliveoil) and help(pdfCluster)
olive1 <- 1 + oliveoil[, 3:10]
margin <- apply(data.matrix(olive1),1,sum)
olive1 <- olive1/margin

# Create some basic visualizations
pairs(olive1, main = "Scatterplot Matrix of Olive Oil Data")

# Define a function to perform t-SNE with different perplexity values and return the result
tsne_model <- function(data, perplexity_value) {
  Rtsne(data, dims = 2, perplexity = perplexity_value)
}

# Tuning exercise: Experiment with different perplexity values
perplexities <- c(3, 30, 60)
results <- lapply(perplexities, function(perplexity) {
  tsne_mod <- tsne_model(olive1, perplexity)
  silhouette_score <- cluster.stats(dist(tsne_mod$Y), cutree(hclust(dist(tsne_mod$Y)), k=3))$avg.silwidth
  return(list(tsne_model = tsne_mod, score = silhouette_score))
})

# Model selection: Find the best t-SNE model based on the chosen metric (e.g., silhouette score)
best_model <- results[[which.max(sapply(results, function(res) res$score))]]

# Plot the output of the t-SNE
plot(best_model$tsne_model$Y, 
     col = oliveoil$macro.area,
     pch = 20, main = "t-SNE Reduction of Olive Oil Data")


