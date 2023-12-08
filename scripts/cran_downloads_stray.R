# Load necessary libraries
library(tidyverse)       # For data manipulation and plotting
library(lubridate)       # For handling date and time objects
library(cranlogs)        # For CRAN downloads data
library(stray)           # For anomaly detection in high-dimensional data

# Define a vector of R packages for analysis
packages <- c("ggplot2", "dplyr", "tidyr", "readr", "purrr", "tibble")

# Function to get CRAN download data for a set of packages
start_date <- as.Date("2022-01-01")
end_date <- today() - 2
pkg_downloads <- cranlogs::cran_downloads(packages = packages, from = start_date, to = end_date)

# Prepare the data in a wide format suitable for high-dimensional analysis
wide_data <- pkg_downloads %>%
  filter(count > 0) %>% 
  pivot_wider(names_from = package, values_from = count)

# Apply stray for anomaly detection
set.seed(123) # For reproducibility
results <- stray::find_HDoutliers(wide_data[,-1],
                                  alpha = 0.2) # Had to change this to get any outliers

# Look at how many anomalies it found
table(results$outliers)

# Extract anomaly indices
anomaly_indices <- results$outliers

# Create some extra columns of the data set
wide_data$anomaly <- FALSE
wide_data$anomaly[results$outliers] = TRUE
wide_data$anomaly_score <- results$out_scores

# Look at the data
wide_data %>% arrange(desc(anomaly_score))

# Now plot the data
longer_data <- wide_data %>% pivot_longer(-c(`date`, `anomaly`, `anomaly_score`),
                            names_to = 'Package',
                            values_to = 'Count')
ggplot(longer_data, aes(x = date, y = Count, colour = Package)) +
  geom_line() + 
  geom_point(data = longer_data %>% filter(anomaly == TRUE)) +
  facet_grid(Package ~ .) +
  theme(legend.position = "None")
  
  
  
