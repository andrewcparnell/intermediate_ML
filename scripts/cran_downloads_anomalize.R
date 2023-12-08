# Load necessary libraries
library(anomalize) # For anomaly detection
library(tidyverse) # For data manipulation and visualization
library(cranlogs)  # For CRAN downloads data

# Set seed for reproducibility
set.seed(123)

# Fetching the tidyverse downloads data - could change to another package
start_date <- as.Date("2022-01-01")
end_date <- today() - 2
pkg_downloads <- cran_downloads(packages = "tidyverse", from = start_date, to = end_date)

# Convert to tibble and arrange by date for consistency
pkg_data <- as_tibble(pkg_downloads) %>%
  arrange(date)

# Initial plot of downloads over time
ggplot(pkg_data, aes(x = date, y = count)) +
  geom_line() +
  labs(title = "Daily Downloads of the tidyverse Package", x = "Date", y = "Download Count")

# Decomposing the time series data into seasonal, trend, and remainder components
decomposed_data <- pkg_data %>%
  time_decompose(count, method = "stl")

# Plotting the decomposed components
pkg_data %>%
  time_decompose(count, method = "stl") %>% 
  anomalize(remainder, method = "iqr") %>% 
  plot_anomaly_decomposition()

# Applying the anomalize function to detect anomalies
anomalized_data <- decomposed_data %>%
  anomalize(remainder)

# Plotting the anomalies
anomalized_data %>%
  time_recompose() %>%
  plot_anomalies(time_recomposed = TRUE, ncol = 3, alpha_dots = 0.5) +
  labs(title = "Anomaly Detection in tidyverse Package Downloads")

# For example, using the gesd method
gesd_anomalized_data <- decomposed_data %>%
  anomalize(remainder, method = "gesd")

# Plotting the results with gesd method
gesd_anomalized_data %>%
  time_recompose() %>%
  plot_anomaly_decomposition() +
  labs(title = "GESD Method for Anomaly Detection in Tidyverse Downloads")

# Extract the dates and counts of anomalies for further analysis
anomaly_dates <- anomalized_data %>%
  filter(anomaly == "Yes") %>%
  select(date, observed, anomaly)

# Print out the anomaly dates and counts
print(anomaly_dates)

