# feature_engineering.ipynb

import pandas as pd
import numpy as np
import pickle

# Load the data saved from previous steps (traffic_data.csv)
#df = pd.read_csv('traffic_data.csv')

# Load the pickled data
with open('tra_X_tr.pkl', 'rb') as f:
    tra_X_tr = pickle.load(f)

with open('tra_Y_tr.pkl', 'rb') as f:
    tra_Y_tr = pickle.load(f)

with open('tra_X_te.pkl', 'rb') as f:
    tra_X_te = pickle.load(f)

with open('tra_Y_te.pkl', 'rb') as f:
    tra_Y_te = pickle.load(f)

# Convert tra_X_tr and tra_Y_tr to dense format if necessary
# Assuming tra_X_tr is a sparse matrix; convert it to a dense DataFrame
df_tra_X_tr = pd.DataFrame(tra_X_tr[0][0].toarray())  # Adjust indices as needed based on structure
df_tra_Y_tr = pd.DataFrame(tra_Y_tr[:, 0], columns=['Traffic_Point_1'])

# Combine features and target into a single DataFrame
df = pd.concat([df_tra_X_tr, df_tra_Y_tr], axis=1)

# Assuming we want to use index or some other column to generate a datetime index
# Example: Creating a timestamp column based on an incremental index if datetime isn't available
df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='15T')  # Adjust the frequency as needed
df.set_index('timestamp', inplace=True)

# Feature Engineering: Extract time-based features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Rolling Averages: Create moving average features for smoothing and trend analysis
df['rolling_mean_traffic'] = df['Traffic_Point_1'].rolling(window=3, min_periods=1).mean()

# Lag Features: Create lagged features for the target variable to capture temporal dependencies
df['lag_1'] = df['Traffic_Point_1'].shift(1)
df['lag_2'] = df['Traffic_Point_1'].shift(2)

# Drop any rows with NaN values created by lagging or rolling operations
df.dropna(inplace=True)

# Save the engineered features to a new CSV for model training
df.to_csv('engineered_features.csv', index=True)

# Display the engineered features and discuss their significance
print("Engineered Features:")
print(df.head())

# Discussion:
# - Hour: Helps model capture daily patterns in traffic flow.
# - Day of Week: Captures weekly patterns, distinguishing weekends from weekdays.
# - Is Weekend: Binary feature to indicate weekends, where traffic patterns differ.
# - Rolling Mean Traffic: Smooths out short-term fluctuations, capturing underlying trends.
# - Lag Features: Enable the model to use past traffic flow values, crucial for time-series forecasting.

# Testing Data Preparation (Optional)
# If needed later for model evaluation, you can prepare testing data similarly.
df_tra_X_te = pd.DataFrame(tra_X_te[0][0].toarray())  # Adjust indices based on actual structure
df_tra_Y_te = pd.DataFrame(tra_Y_te[:, 0], columns=['Traffic_Point_1'])

# Combine testing features and target if needed for future evaluation
df_test = pd.concat([df_tra_X_te, df_tra_Y_te], axis=1)

# Optional: Save testing features for evaluation (if you plan to use them later)
df_test.to_csv('engineered_test_features.csv', index=False)  # Saving for potential future use