# Real-Time Traffic Prediction with Kafka - Starter Notebook

# -----------------------------------------
# Kafka Installation Guide (with Zookeeper)
# -----------------------------------------

# Step 1: Download Kafka
# ----------------------
# 1. Go to the official Kafka website: https://kafka.apache.org/downloads
# 2. Download the latest stable version that supports Scala 2.13:
#    - Kafka Version: kafka_2.13-3.8.0 (or the latest stable version)
# 3. Unzip the downloaded file into your preferred installation directory:
#    Example:
#    tar -xzf kafka_2.13-3.8.0.tgz

# Step 2: Start Zookeeper
# -----------------------
# Kafka requires Zookeeper for cluster management. 
# You can start Zookeeper using the following command:
# Navigate to your Kafka installation directory and run:
#    bin/zookeeper-server-start.sh config/zookeeper.properties NO!
#    WINDOWS VERSION
#    .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
# This starts Zookeeper on the default port (2181). Keep this terminal running.

# Step 3: Start Kafka Broker
# --------------------------
# After Zookeeper is up and running, start Kafka broker using:
#    bin/kafka-server-start.sh config/server.properties
#    WINDOWS VERSION
#    .\bin\windows\kafka-server-start.bat .\config\server.properties
# This starts the Kafka broker on the default port (9092). Keep this terminal running as well.

# Step 4: Create a Kafka Topic (for testing)
# ------------------------------------------
# Once Kafka is running, you can create a topic to send and consume data:
#    bin/kafka-topics.sh --create --topic traffic_data --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

# Step 5: Check Topic
# -------------------
# You can check if the topic was created successfully by running:
#    bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
# The "traffic_data" topic should appear in the list.

# Step 6: Start Kafka Console Producer and Consumer (Optional for Testing)
# ------------------------------------------------------------------------
# You can use Kafka's console producer and consumer tools to manually send and receive messages for testing:
# 1. Console Producer:
#    bin/kafka-console-producer.sh --broker-list localhost:9092 --topic traffic_data
# 2. Console Consumer:
#    bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic traffic_data --from-beginning
# These tools are useful to verify that Kafka is working correctly before running your Python scripts.

# -----------------------------------------------
# Data Preparation, Producer and Consumer Scripts
# -----------------------------------------------

# Importing Necessary Libraries
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.sparse import csr_matrix
import gzip
import json

# Phase 1: Data Preparation

# Step 1: Load the Traffic Flow Forecasting Dataset from the .mat File
import scipy.io

# Dummy data for testing TROUBLESHOOTING JUST DELETE LATER

# dummy_series = pd.Series(np.random.randn(100))  # 100 random data points

# plot_acf(dummy_series, lags=50)
# plt.show()

## COMMENT OUT DUMMY DATA ONCE GOOD DATA IS GOOD

# Function to load and prepare the data
def load_and_prepare_data(mat_file_path):
    # Load the .mat file
    mat = scipy.io.loadmat(mat_file_path)
    
    # Extract training and testing data
    tra_X_tr = mat['tra_X_tr']
    tra_Y_tr = mat['tra_Y_tr']
    tra_X_te = mat['tra_X_te']
    tra_Y_te = mat['tra_Y_te']
    tra_adj_mat = mat['tra_adj_mat']
    
    # Save the data using pickle for later use
    with open('tra_X_tr.pkl', 'wb') as f:
        pickle.dump(tra_X_tr, f)

    with open('tra_Y_tr.pkl', 'wb') as f:
        pickle.dump(tra_Y_tr, f)

    print("Data preparation complete. Data saved as pickle files.")

# Load and prepare the data (You need to replace the file path with the actual path to your .mat file)
load_and_prepare_data('C:\\Users\\arian\\.0.OPAI\\Individual Assignment\\traffic_dataset.mat') ##path_to_traffic_dataset.mat

# ---------------------------------------------
# Guide for Creating Kafka Producer and Consumer
# ---------------------------------------------

# Phase 2: Kafka Producer and Consumer
# Guide:
# In this phase, you need to create two Python scripts: one for the producer and one for the consumer.
# The producer will read the data and send it to a Kafka topic, and the consumer will consume it in real-time.

# Create a producer script (producer.py) that sends the data in small chunks to Kafka.
# Create a consumer script (consumer.py) that listens to the Kafka topic and processes the messages.
# You can use the partial code snippets below as a guide.

# Producer Script Example (Partial)

#'''
from kafka import KafkaProducer
import pickle
import json
import gzip
#import time #ADDED

# Initialize Kafka Producer
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Load data and send it in small chunks (adjust chunk size as needed)
# You will implement this as shown in the producer.py script
# # Load the pickled data ADDED
# with open('tra_X_tr.pkl', 'rb') as f:
#     tra_X_tr = pickle.load(f)

# with open('tra_Y_tr.pkl', 'rb') as f:
#     tra_Y_tr = pickle.load(f)
#     ##ADDED

# # Convert data to pandas DataFrames for easier handling
# df_tra_X_tr = pd.DataFrame(tra_X_tr[0][0].toarray())  # Convert sparse matrix to dense
# df_tra_Y_tr = pd.DataFrame(tra_Y_tr[:, 0])


compressed_data = gzip.compress(json.dumps(data).encode('utf-8')) ## ADDED CAN COMMENT OUT IF PROBS

producer.send('traffic_data', compressed_data)
# '''

# # Consumer Script Example (Partial)
# '''
from kafka import KafkaConsumer
import gzip
import json

# Initialize Kafka Consumer
consumer = KafkaConsumer('traffic_data', bootstrap_servers='localhost:9092')

# Consume and process messages from Kafka topic
# You will implement this as shown in the consumer.py script

# '''

# ----------------------------------------
# Exploratory Data Analysis (EDA) Example
# ----------------------------------------

# Load the training data from the pickle files
with open('tra_X_tr.pkl', 'rb') as f:
    tra_X_tr = pickle.load(f)

with open('tra_Y_tr.pkl', 'rb') as f:
    tra_Y_tr = pickle.load(f)

# Convert data to pandas DataFrame for easier manipulation
df_tra_X_tr = pd.DataFrame(tra_X_tr[0][0].toarray())  # Convert sparse matrix to dense
df_tra_Y_tr = pd.DataFrame(tra_Y_tr[:, 0])

# Step 1: Visualize Traffic Flow Over Time
# Let's assume we're visualizing the traffic flow for the first location
location_index = 0
traffic_flow_series = df_tra_Y_tr.iloc[location_index, :]

print(traffic_flow_series) ##ADDED FOR TROUBLE SHOOT FEEL FREE TO REMOVE
print(f"Length of traffic_flow_series: {len(traffic_flow_series)}") ##ADDED FOR TROUBLE SHOOT FEEL FREE TO REMOVE

plt.figure(figsize=(10, 5))
plt.plot(traffic_flow_series)
plt.title(f"Traffic Flow Over Time at Location {location_index + 1}")
plt.xlabel("Time (15-minute intervals)")
plt.ylabel("Traffic Flow")
plt.show()

# Step 2: Autocorrelation and Partial Autocorrelation
plot_acf(traffic_flow_series, lags=50)
plt.title(f"Autocorrelation Function (ACF) for Location {location_index + 1}")
plt.show()

plot_pacf(traffic_flow_series, lags=50)
plt.title(f"Partial Autocorrelation Function (PACF) for Location {location_index + 1}")
plt.show()

# Guide:
# In this phase, you should further explore the time-series nature of the data by generating more visualizations.
# You may want to explore different locations, investigate daily or weekly seasonality, or look at speed and occupancy data if available.

# ----------------------------------------
# Feature Engineering and Model Training
# ----------------------------------------

# Feature Engineering Example
rolling_window_size = 4  # Rolling average over 1 hour (4 intervals)
df_tra_X_tr['rolling_mean'] = df_tra_X_tr.iloc[:, 0].rolling(window=rolling_window_size).mean()

# Fill any NaN values resulting from the rolling operation
df_tra_X_tr.fillna(method='bfill', inplace=True)

# Train a Simple Linear Regression Model
X_train = df_tra_X_tr[['rolling_mean']]  # Example feature
y_train = df_tra_Y_tr.values  # Assuming this is the target variable

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_train_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, y_train_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"Training MAE: {mae}")
print(f"Training RMSE: {rmse}")

# Guide:
# Add more features, try different models, and integrate Kafka producer and consumer scripts into your pipeline.
