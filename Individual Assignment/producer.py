# producer.py

import pickle
from kafka import KafkaProducer
import json
import time
import gzip
import sys
import pandas as pd
from scipy.sparse import csr_matrix

# Initialize Kafka Producer
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# TROUBLE SHOOTING
# producer.send('test_topic', b'test_message')
# producer.flush()

# Load the pickled data
with open('tra_X_tr.pkl', 'rb') as f:
    tra_X_tr = pickle.load(f)

with open('tra_Y_tr.pkl', 'rb') as f:
    tra_Y_tr = pickle.load(f)

with open('tra_X_te.pkl', 'rb') as f:
    tra_X_te = pickle.load(f)

with open('tra_Y_te.pkl', 'rb') as f:
    tra_Y_te = pickle.load(f)

# Convert to pandas DataFrames for easier handling (if not already)
df_tra_X_tr = pd.DataFrame(tra_X_tr[0][0].toarray())  # Convert sparse matrix to dense
df_tra_Y_tr = pd.DataFrame(tra_Y_tr[:, 0])

# Define the chunk size (e.g., split data into groups of 10 locations)
chunk_size = 10
num_locations = df_tra_Y_tr.shape[0]  # Should be 36

def send_traffic_data():
    # Loop through each time step in the training data and send it to Kafka
    for i in range(df_tra_Y_tr.shape[1]):
        for start in range(0, num_locations, chunk_size):
            end = min(start + chunk_size, num_locations)
            # Example data payload for a subset of locations
            data = {
                'time_step': i,
                'location_data': df_tra_Y_tr.iloc[start:end, i].values.tolist(),  # Convert DataFrame slice to list
                'features': df_tra_X_tr.iloc[start:end, :].values.tolist()  # Convert DataFrame slice to list
            }
            # Compress the data before sending
            compressed_data = gzip.compress(json.dumps(data).encode('utf-8'))

            # Check the size of the compressed data
            size_of_message = sys.getsizeof(compressed_data)
            print(f"Size of compressed message at time step {i}, locations {start}-{end}: {size_of_message} bytes")

            # Ensure the size does not exceed 1 GB (or other appropriate limit)
            if size_of_message > 1073741824:  # 1 GB in bytes
                print("Warning: Message size exceeds 1 GB. Consider reducing chunk size or further compressing data.")
                continue
            
            # Send compressed data to Kafka topic 'traffic_data'
            producer.send('traffic_data', compressed_data)
            time.sleep(1)  # Adjust delay as needed for smaller chunks

if __name__ == "__main__":
    send_traffic_data()
    print("Data sending complete.")

################################### CONSUMER #############################################
import kafka
from kafka import KafkaConsumer
import gzip
import json
import pandas as pd
import matplotlib.pyplot as plt

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    'traffic_data',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',  # Start reading at the earliest message in the topic
    enable_auto_commit=True,  # Automatically commit offsets
    group_id='traffic-consumer-group'  # Group ID for managing offsets
)

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

def consume_traffic_data():
    print("Starting to consume messages...")
    traffic_data = []
    for message in consumer:
        try:
            # Attempt to decompress the message
            try:
                decompressed_data = gzip.decompress(message.value).decode('utf-8')
            except gzip.BadGzipFile:
                # If not gzipped, assume it's plain JSON
                decompressed_data = message.value.decode('utf-8')
            
            data = json.loads(decompressed_data)
            
            # Extract the data
            time_step = data['time_step']
            location_data = data['location_data']
            features = data['features']
            
            # Convert the received lists back to DataFrames for further processing if needed
            df_location_data = pd.DataFrame(location_data)
            df_features = pd.DataFrame(features, columns = column_headers)
            
            print(f"Received data for time step {time_step} with {df_location_data.shape[0]} locations")
            print(f"Location Data Head:\n{df_location_data.head()}")
            print(f"Features Data Head:\n{df_features.head()}")

            traffic_data.append(df_features)
            
        except Exception as e:
            print(f"Failed to process message: {e}")

    return traffic_data

if __name__ == "__main__":
    consume_traffic_data()
    print("Data consumption complete.")




