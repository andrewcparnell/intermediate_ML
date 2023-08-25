# Load required libraries
library(kernlab)
library(pdfCluster)

# Load the olive oil data
data(oliveoil)
olive1 <- 1 + oliveoil[, 3:10]
margin <- apply(data.matrix(olive1),1,sum)
olive1 <- olive1/margin
alr <- (-log( olive1[, -4]/olive1[, 4]))

# Basic visualization of the first two variables
# plot(olive1$palmitic, olive1$arachidic, main="Olive Oil: Area vs. Palmitic Acid Content", xlab="Area", ylab="Palmitic Acid Content")

# Perform spectral clustering with 3 clusters
initial_clustering <- specc(as.matrix(olive1), centers = 3)
initial_clusters <- initial_clustering@.Data

# Visualize the clustering result
plot(olive1$palmitic, olive1$arachidic, col = initial_clusters, 
     main="Initial Clustering: Arachidic vs Palmitic", xlab="Palmitic", ylab="Arachidic")

# Tuning exercise: finding the optimal number of clusters
d <- dist(as.matrix(olive1))
tune_spectral <- function(k) {
  clustering <- specc(as.matrix(olive1), centers = k)
  clustering_objective <- cluster.stats(d, clustering@.Data)$avg.silwidth
  return(clustering_objective)
}

k_values <- 2:10
objectives <- sapply(k_values, tune_spectral)

# Plot the tuning result
plot(k_values, objectives, type = "b", main="Tuning Spectral Clustering: Objective vs. Number of Clusters", xlab="Number of Clusters", ylab="Objective")

# Perform clustering with the optimal number of clusters (e.g., 4)
final_clustering <- specc(olive1, centers = 4)
final_clusters <- final_clustering@.Data

# Visualize the final clustering result
plot(olive1$palmitic, olive1$arachidic, col = final_clusters, 
     main="Final Clustering: Arachidic vs Palmitic", 
     xlab="Palmitic", ylab="Arachidic")
