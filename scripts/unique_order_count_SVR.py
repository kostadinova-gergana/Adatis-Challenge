import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
df = pd.read_csv('unique_order_count.csv')
df['Year Month'] = pd.to_datetime(df['Year Month'])
df.set_index('Year Month', inplace=True)

# Sort the DataFrame by the 'Year Month' index
df.sort_index(inplace=True)

# Feature engineering - Transform the Date column
df['MonthsSinceStart'] = (df.index - df.index.min()).days // 30  # Assuming 30 days per month

# Define cutoff date for splitting
cutoff_date = df.index.max() - pd.DateOffset(months=3)

# Split data into training and testing sets
train_df = df[df.index < cutoff_date]
test_df = df[df.index >= cutoff_date]

# Prepare data for training
X_train, y_train = train_df[['MonthsSinceStart']], train_df['Unique Order Count']
X_test, y_test = test_df[['MonthsSinceStart']], test_df['Unique Order Count']

# Step 6: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = SVR(kernel='rbf')
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Calculate errors
errors_df = test_df[['Unique Order Count']].copy()
errors_df['pred_sales'] = predictions
errors_df['errors'] = predictions - y_test
errors_df.insert(0, 'model', 'SVR')

# Print errors DataFrame
print(errors_df)

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(errors_df.index, errors_df['errors'], label='Errors')
ax.plot(errors_df.index, errors_df['Unique Order Count'], label='Average Data')
ax.plot(errors_df.index, errors_df['pred_sales'], label='Forecast')
ax.legend(loc='best')
ax.set_xlabel('Date')
ax.set_ylabel('Average Order Amount')
ax.set_title('SVR forecasts with sales growth and errors')
plt.show()

# Calculate MAE, RMSE, and MAPE
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

# Print the result metrics
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)

# Generate a forecast for the next 3 months
last_date = df.index.max()
last_day_of_last_month = pd.Timestamp(last_date.year, last_date.month, 1) + pd.offsets.MonthEnd(1)
first_day_of_next_month = last_day_of_last_month + pd.Timedelta(days=1)
forecast_dates = pd.date_range(start=first_day_of_next_month, periods=3, freq='MS')
months_since_start_forecast = (forecast_dates - df.index.min()).days // 30  # Assuming 30 days per month
X_forecast_scaled = scaler.transform(months_since_start_forecast.values.reshape(-1, 1))
forecast_predictions = model.predict(X_forecast_scaled)

# Display the forecast results
plt.plot(df.index, df['Unique Order Count'], label='Actual Data')
plt.plot(df.index[df.index < cutoff_date], model.predict(X_train_scaled), color='red', label='Training Predictions')
plt.plot(forecast_dates, forecast_predictions, color='green', label='Forecast for Next 3 Months')
plt.title('Sales Growth Forecasting with Support Vector Regression')
plt.xlabel('Date')
plt.ylabel('Sales Growth')
plt.legend() 
plt.grid(True)
plt.show() 

# Print the forecast values
forecast_df = pd.DataFrame({'MonthsSinceStart': forecast_dates, 'Forecasted_Order_Count': forecast_predictions})
print(forecast_df)

# Reset the index of the original DataFrame
df.reset_index(inplace=True)

# Create a new DataFrame
new_df = df.copy()

# Drop unnecessary columns
new_df.drop(columns=['Month Number', 'Year', 'MonthsSinceStart'], inplace=True)

# Add a new column "Type" and populate it with "Historical Data"
new_df['Type'] = 'Historical Data'

# Append forecasted data
forecast_df = pd.DataFrame({'Year Month': forecast_dates, 'Unique Order Count': forecast_predictions})
forecast_df['Type'] = 'Predicted Data'

# Reset index of forecast_df
forecast_df.reset_index(drop=True, inplace=True)

# Concatenate historical and forecasted data
combined_df = pd.concat([new_df, forecast_df])

# Convert 'Year Month' to datetime type
combined_df['Year Month'] = pd.to_datetime(combined_df['Year Month'])

# Drop the existing index column
combined_df.reset_index(drop=True, inplace=True)

# Export the new data to a CSV file
combined_df.to_csv('unique_order_count_forecast.csv', index=False)