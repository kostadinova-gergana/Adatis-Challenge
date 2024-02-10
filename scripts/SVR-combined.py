import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df['MonthsSinceStart'] = (df.index - df.index.min()).days // 30  # Assuming 30 days per month
    return df

def train_and_predict(df, target_column):
    cutoff_date = df.index.max() - pd.DateOffset(months=3)
    train_df = df[df.index < cutoff_date]
    test_df = df[df.index >= cutoff_date]

    X_train, y_train = train_df[['MonthsSinceStart']], train_df[target_column]
    X_test, y_test = test_df[['MonthsSinceStart']], test_df[target_column]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel='rbf')
    model.fit(X_train_scaled, y_train)

    forecast_dates = pd.date_range(start=test_df.index.max() + pd.DateOffset(days=1), periods=3, freq='MS')
    months_since_start_forecast = (forecast_dates - df.index.min()).days // 30
    X_forecast_scaled = scaler.transform(months_since_start_forecast.values.reshape(-1, 1))
    forecast_predictions = model.predict(X_forecast_scaled)

    forecast_df = pd.DataFrame({'Year Month': forecast_dates, target_column: forecast_predictions})
    forecast_df['Type'] = 'Predicted Data'

    combined_df = pd.concat([df, forecast_df])
    combined_df['Year Month'] = combined_df.index  # For historical data, use the index as the date
    combined_df.loc[combined_df['Type'] == 'Predicted Data', 'Year Month'] = forecast_dates  # For predicted data, use forecasted dates
    combined_df['Type'] = combined_df['Type'].fillna('Historical Data')  # Populate 'Type' column for historical data
    combined_df = combined_df[['Year Month', target_column, 'Type']]  # Rearranging columns
    
    return combined_df.reset_index(drop=True)  # Resetting index and dropping the old index column

# Load and preprocess data for average order amount
df_order_amount_avg = pd.read_csv('C:/Users/geri0/SynologyDrive/Магистър/Курсови работи/Семестър 1/Управление на бизнес модели/final-source/order_amount_avg.csv')
df_order_amount_avg['Year Month'] = pd.to_datetime(df_order_amount_avg['Year Month'])
df_order_amount_avg.set_index('Year Month', inplace=True)
df_order_amount_avg = preprocess_data(df_order_amount_avg)
forecasted_order_amount_avg = train_and_predict(df_order_amount_avg, 'AVG Order Amount GBP')

# Load and preprocess data for online sales percentage
df_online_sales = pd.read_csv('C:/Users/geri0/SynologyDrive/Магистър/Курсови работи/Семестър 1/Управление на бизнес модели/final-source/online_sales_percentage.csv')
df_online_sales['Year Month'] = pd.to_datetime(df_online_sales['Year Month'])
df_online_sales.set_index('Year Month', inplace=True)
df_online_sales = preprocess_data(df_online_sales)
forecasted_online_sales_percentage = train_and_predict(df_online_sales, 'Online Sales Percentage')

# Load and preprocess data for average order value
df_order_value_avg = pd.read_csv('C:/Users/geri0/SynologyDrive/Магистър/Курсови работи/Семестър 1/Управление на бизнес модели/final-source/order_frequency.csv')
df_order_value_avg['Year Month'] = pd.to_datetime(df_order_value_avg['Year Month'])
df_order_value_avg.set_index('Year Month', inplace=True)
df_order_value_avg = preprocess_data(df_order_value_avg)
forecasted_order_frequency = train_and_predict(df_order_value_avg, 'Order Frequency')

# Load and preprocess data for unique order count
df_unique_order_count = pd.read_csv('C:/Users/geri0/SynologyDrive/Магистър/Курсови работи/Семестър 1/Управление на бизнес модели/final-source/unique_order_count.csv')
df_unique_order_count['Year Month'] = pd.to_datetime(df_unique_order_count['Year Month'])
df_unique_order_count.set_index('Year Month', inplace=True)
df_unique_order_count = preprocess_data(df_unique_order_count)
forecasted_unique_order_count = train_and_predict(df_unique_order_count, 'Unique Order Count')

# Load and preprocess data for units per transaction
df_units_per_transaction = pd.read_csv('C:/Users/geri0/SynologyDrive/Магистър/Курсови работи/Семестър 1/Управление на бизнес модели/final-source/units_per_order.csv')
df_units_per_transaction['Year Month'] = pd.to_datetime(df_units_per_transaction['Year Month'])
df_units_per_transaction.set_index('Year Month', inplace=True)
df_units_per_transaction = preprocess_data(df_units_per_transaction)
forecasted_units_per_order = train_and_predict(df_units_per_transaction, 'Units Per Order')

# Storing the combined DataFrames in separate dictionaries
combined_dfs = {
    'online_sales_percentage': forecasted_online_sales_percentage,
    'order_amount_avg': forecasted_order_amount_avg,
    'order_frequency': forecasted_order_frequency,
    'unique_order_count': forecasted_unique_order_count,
    'units_per_order': forecasted_units_per_order
}

print(combined_dfs)
