# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from kafka import KafkaConsumer
import gzip
import json

# Step 1: Consume and Store Data Locally
# Define Kafka Consumer settings
consumer = KafkaConsumer(
    'traffic_data',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='traffic-consumer-group'
)

# Define the column headers based on your Kafka data structure
column_headers = [
    'Traffic_Point_1', 'Traffic_Point_2', 'Traffic_Point_3', 'Traffic_Point_4', 
    'Traffic_Point_5', 'Traffic_Point_6', 'Traffic_Point_7', 'Traffic_Point_8', 
    'Traffic_Point_9', 'Traffic_Point_10',
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
    'Hour_0', 'Hour_1', 'Hour_2', 'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6', 'Hour_7',
    'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12', 'Hour_13', 'Hour_14', 'Hour_15',
    'Hour_16', 'Hour_17', 'Hour_18', 'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23',
    'Direction_North', 'Direction_South', 'Direction_East', 'Direction_West',
    'Number_of_Lanes', 'Road_Name', 'Extra_Column'
]

# Initialize a list to store consumed data
traffic_data = []

# Consume the data from Kafka
for message in consumer:
    try:
        # Decompress the message if it is gzipped
        try:
            decompressed_data = gzip.decompress(message.value).decode('utf-8')
        except gzip.BadGzipFile:
            decompressed_data = message.value.decode('utf-8')

        # Convert JSON string to dictionary
        data = json.loads(decompressed_data)

        # Convert lists to DataFrame format
        df_features = pd.DataFrame(data['features'], columns=column_headers)

        # Append to traffic data list
        traffic_data.append(df_features)

        # For demonstration, stop after a certain number of messages
        if len(traffic_data) >= 100:  # Adjust the number as needed
            break

    except Exception as e:
        print(f"Error processing message: {e}")

# Combine all received data into a single DataFrame
df_traffic = pd.concat(traffic_data, ignore_index=True)

# Step 2: Visualizations
# Time-Series Plot for Traffic Flow vs. Time
plt.figure(figsize=(15, 6))
sns.lineplot(data=df_traffic.filter(regex='Traffic_Point'), dashes=False)
plt.title('Traffic Flow vs. Time')
plt.xlabel('Time')
plt.ylabel('Traffic Flow')
plt.grid(True)
plt.show()

# Autocorrelation and Partial Autocorrelation Plots
# Example using Traffic_Point_1 column for demonstration
plt.figure(figsize=(12, 6))
plot_acf(df_traffic['Traffic_Point_1'], lags=50)
plt.title('Autocorrelation of Traffic Flow (Traffic_Point_1)')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(df_traffic['Traffic_Point_1'], lags=50)
plt.title('Partial Autocorrelation of Traffic Flow (Traffic_Point_1)')
plt.grid(True)
plt.show()

# Additional Visualizations: Heatmap of Feature Correlations
plt.figure(figsize=(15, 10))
sns.heatmap(df_traffic.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()

# Save Data for Further Analysis
df_traffic.to_csv('traffic_data.csv', index=False)