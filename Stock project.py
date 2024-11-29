import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv(""C:\Users\HP\Desktop\Stock (1).csv"")

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values(by='Date')

# Set the Date column as the index
df.set_index('Date', inplace=True)

# Get the target year from the user
target_year = int(input("Enter the target year for prediction (e.g., 2024): "))
inflation_rate = 0.03
df['Year'] = df.index.year
years_to_inflate = target_year - df['Year']

# Adjust prices for inflation
for column in ['Open', 'High', 'Low', 'Close']:
    df[column] = df[column] * ((1 + inflation_rate) ** years_to_inflate)

# Prepare data for modeling
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create the dataset for Random Forest
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 1  # For Random Forest
X, y = create_dataset(scaled_data, time_step)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Make predictions on the training data
rf_predictions = rf_model.predict(X)

# Calculate MSE and R² for Random Forest
rf_mse = mean_squared_error(y, rf_predictions)
rf_r2 = r2_score(y, rf_predictions)

# Print the results
print("\nRandom Forest Results:")
print(f'Mean Squared Error: {rf_mse:.4f}')
print(f'R² Score: {rf_r2:.4f}\n')

# Visualize historical data and predictions
plt.figure(figsize=(14, 7))

# Plot actual prices
plt.plot(df.index, df['Close'], color='blue', label='Actual Stock Price')

# Plot Random Forest predictions
plt.plot(df.index[time_step+1:], scaler.inverse_transform(rf_predictions.reshape(-1, 1)), color='red', label='Random Forest Predicted Price')

plt.title('Historical Stock Price and Prediction')
plt.xlabel('Year')
plt.ylabel('Stock Price')

# Set x-ticks to show years from start to end of data
start_year = df.index.year.min()
end_year = df.index.year.max()
plt.xticks(pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='YS'), 
           range(start_year, end_year + 1), rotation=45)

plt.legend()
plt.tight_layout()
plt.show()

# Predict for the target year
last_data = scaled_data[-time_step:].reshape(1, -1)
future_predictions_rf = []

# Predicting for the target year (next 12 months)
for month in range(12):
    rf_next_pred = rf_model.predict(last_data)
    future_predictions_rf.append(rf_next_pred[0])  
    last_data = np.append(last_data[0][1:], rf_next_pred).reshape(1, -1)

# Inverse transform the predictions
future_predictions_rf = scaler.inverse_transform(np.array(future_predictions_rf).reshape(-1, 1))

# Create a future date range for the entire target year
future_dates = pd.date_range(start=pd.to_datetime(f'{target_year}-01-01'), end=pd.to_datetime(f'{target_year}-12-31'), freq='M')

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Stock Price': future_predictions_rf.flatten()
})

# Display the predictions
print(f"\nPredicted Stock Prices for {target_year}:")
print(predictions_df.to_string(index=False))

# Calculate and display the average predicted stock price
avg_prediction = predictions_df['Predicted Stock Price'].mean()
print(f"\nAverage Predicted Stock Price for {target_year}: ${avg_prediction:.2f}")

# Plot future predictions
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], color='blue', label='Historical Stock Price')
plt.plot(future_dates, future_predictions_rf, color='orange', marker='o', label=f'RF Predicted Stock Price for {target_year}')
plt.title(f'Stock Price Predictions for {target_year}\nFinal Predicted Price: {future_predictions_rf[-1][0]:.2f}')
plt.xlabel('Year')
plt.ylabel('Stock Price')

# Set x-ticks to show years from start of data to target year
start_year = df.index.year.min()
plt.xticks(pd.date_range(start=f'{start_year}-01-01', end=f'{target_year}-12-31', freq='YS'), 
           range(start_year, target_year + 1), rotation=45)

plt.legend()
plt.tight_layout()
plt.show()
