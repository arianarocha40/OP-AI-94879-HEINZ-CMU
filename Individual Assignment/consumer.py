import kafka
from kafka import KafkaConsumer
import gzip
import json
import pandas as pd

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
            df_features = pd.DataFrame(features, column = column_headers)
            
            print(f"Received data for time step {time_step} with {df_location_data.shape[0]} locations")
            print(f"Location Data Head:\n{df_location_data.head()}")
            print(f"Features Data Head:\n{df_features.head()}")
            
        except Exception as e:
            print(f"Failed to process message: {e}")

if __name__ == "__main__":
    consume_traffic_data()
    print("Data consumption complete.")
