# Load necessary libraries
library(tidyverse)       # For data manipulation and plotting
library(lubridate)       # For handling date and time objects
library(cranlogs)        # For CRAN downloads data
library(tsoutliers)      # For detecting outliers in time-series data
library(forecast)        # For time series forecasting

# Get some data
start_date <- as.Date("2023-10-01")
end_date <- today() - 2
pkg_downloads <- cran_downloads(packages = "Rcpp", from = start_date, to = end_date)

# Convert to tibble and arrange by date for consistency
pkg_data <- as_tibble(pkg_downloads) %>%
  arrange(date)

# Convert to a ts object for time series analysis
ts_data <- ts(pkg_data$count, frequency = 7) # Could also use 365
plot(ts_data)

# Use the tsoutliers function to detect outliers
outliers <- tsoutliers::tso(ts_data)
# Quite slow

# Print summary of model fitted and outliers detected
print(outliers)

# Plot original data with outliers highlighted
plot(outliers)

# Adjust the time series data by removing the detected outliers
adjusted_ts_data <- outliers$yadj

# Plot adjusted time series
plot(adjusted_ts_data, main = "Adjusted Time Series after Removing Outliers")

# Re-forecast the time series using the adjusted data
forecast_model <- forecast::auto.arima(adjusted_ts_data)
forecasted_data <- forecast(forecast_model)

# Plot the forecast
plot(forecasted_data, main = "Forecast Using Adjusted Time Series")

