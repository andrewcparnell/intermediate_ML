# Perform some UMAP dimension reduction on the olive oil data
# See the vignette at https://cran.r-project.org/web/packages/umap/vignettes/umap.html
# for more detail

# Load necessary libraries
library(pdfCluster)
library(umap)
library(ggplot2)
library(fpc)

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

# Define a function to perform UMAP with different hyperparameters and return a chosen metric (e.g., silhouette score)
# watch out - American spellings!
umap_model <- function(data, neighbors, distance_metric) {
  custom_config <- umap.defaults
  custom_config$metric <- distance_metric
  custom_config$n_neighbors <- as.numeric(neighbors)
  umap_result <- umap(data, config = custom_config)
  silhouette_score <- cluster.stats(dist(umap_result$layout), cutree(hclust(dist(umap_result$layout)), k=3))$avg.silwidth
  return(list(umap = umap_result, score = silhouette_score))
}

# Hyperparameter tuning: Experiment with different numbers of neighbors and distance metrics
neighbors <- c(5, 10, 15)
distance_metrics <- c("euclidean", "manhattan", "cosine")
models <- expand.grid(neighbors = neighbors, distance_metrics = distance_metrics)

results <- apply(models, 1, function(params) {
  umap_model(olive1, params['neighbors'], params['distance_metrics'])
})

# Plot some of the models
plot(results[[1]]$umap$layout, col = oliveoil$macro.area)
plot(results[[9]]$umap$layout, col = oliveoil$macro.area)

# Model selection: Find the best UMAP model based on the chosen metric (e.g., silhouette score)
best_model <- results[[which.max(sapply(results, function(res) res$score))]]

# Plot the output of the best model
plot(best_model$umap$layout, col = oliveoil$macro.area)
plot(best_model$umap$layout, col = oliveoil$region)

# New values and prediction
new_data <- olive1[1:50,] + matrix(rnorm(50*8, 0, 0.1), 
                                   ncol = 8)
new_data_umap <- predict(best_model$umap, new_data)
points(new_data_umap, col = 'blue', pch = 19)


