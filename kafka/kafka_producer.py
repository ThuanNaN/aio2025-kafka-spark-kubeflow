"""
Kafka Producer with SDV Stream Generator
Generates synthetic heart disease data and streams to Kafka topic
"""

# Fix OpenMP duplicate library issue on Windows
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import time
import json
from datetime import datetime
from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError, KafkaError
from kafka_config import KafkaConfig

class KafkaStreamProducer:
    def __init__(self, config_module=None):
        """Initialize Kafka Producer with stream generator"""
        self.config = config_module or KafkaConfig()
        self.producer = None
        self.synthesizer = None
        self.original_data = None
        self.total_sent = 0
        
    def connect_kafka(self):
        """Connect to Kafka and create topic if not exists"""
        print("\n" + "=" * 70)
        print("KAFKA CONNECTION")
        print("=" * 70)
        
        try:
            # Create Kafka producer
            print(f"\n[1] Connecting to Kafka at {self.config.BOOTSTRAP_SERVERS}...")
            self.producer = KafkaProducer(**self.config.PRODUCER_CONFIG)
            print("Successfully connected to Kafka broker")
            
            # Create topic manually if not exists
            print(f"\n[2] Checking topic '{self.config.TOPIC_NAME}'...")
            try:
                admin_client = KafkaAdminClient(
                    bootstrap_servers=self.config.BOOTSTRAP_SERVERS,
                    client_id='producer_admin',
                    api_version=(2, 8, 0)
                )
                
                topic = NewTopic(
                    name=self.config.TOPIC_NAME,
                    num_partitions=self.config.NUM_PARTITIONS,
                    replication_factor=self.config.REPLICATION_FACTOR
                )
                
                try:
                    admin_client.create_topics([topic], validate_only=False)
                    print(f"Created topic '{self.config.TOPIC_NAME}'")
                except TopicAlreadyExistsError:
                    print(f"Topic '{self.config.TOPIC_NAME}' already exists")
                except Exception as e:
                    print(f"Note: {e}")
                    print("Topic will be auto-created on first message")
                
                admin_client.close()
            except Exception as e:
                print(f"Could not create topic manually: {e}")
                print("Topic will be auto-created on first message (auto-create enabled)")
            
            return True
            
        except Exception as e:
            print(f"Error connecting to Kafka: {e}")
            print("\nMake sure Kafka is running:")
            print("  docker-compose up -d")
            return False
    
    def load_and_train_model(self, data_path, categorical_columns):
        """Load data and train SDV model"""
        print("\n" + "=" * 70)
        print("SDV MODEL TRAINING")
        print("=" * 70)
        
        print(f"\n[1] Loading dataset from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.original_data = pd.read_csv(data_path)
        print(f"Loaded {len(self.original_data)} records with {len(self.original_data.columns)} columns")
        
        # Create metadata
        print("\n[2] Creating metadata...")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.original_data)
        
        for col in categorical_columns:
            if col in self.original_data.columns:
                metadata.update_column(column_name=col, sdtype='categorical')
        
        print(f"Configured {len(categorical_columns)} categorical columns")
        
        # Train model
        print("\n[3] Training GaussianCopulaSynthesizer...")
        self.synthesizer = GaussianCopulaSynthesizer(
            metadata=metadata,
            enforce_min_max_values=True,
            enforce_rounding=True
        )
        
        self.synthesizer.fit(self.original_data)
        print("Model training completed!")
    
    def generate_batch(self, batch_size):
        """Generate synthetic data batch"""
        synthetic_data = self.synthesizer.sample(num_rows=batch_size)
        return synthetic_data
    
    def send_to_kafka(self, data, batch_number):
        """Send data batch to Kafka topic"""
        records_sent = 0
        
        for index, row in data.iterrows():
            try:
                # Create message
                message = {
                    'batch_number': batch_number,
                    'record_id': self.total_sent + records_sent,
                    'timestamp': datetime.now().isoformat(),
                    'data': row.to_dict()
                }
                
                # Serialize to JSON
                message_json = json.dumps(message)
                
                # Send to Kafka
                future = self.producer.send(
                    self.config.TOPIC_NAME,
                    value=message_json,
                    key=str(message['record_id'])
                )
                
                # Wait for confirmation (optional, can be async)
                future.get(timeout=10)
                records_sent += 1
                
            except KafkaError as e:
                print(f"Error sending record {records_sent}: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
        
        # Ensure all messages are sent
        self.producer.flush()
        self.total_sent += records_sent
        
        return records_sent
    
    def stream_continuous(self, batch_size=10, interval_seconds=2):
        """Stream data continuously"""
        print("\n" + "=" * 70)
        print("CONTINUOUS STREAMING TO KAFKA")
        print("=" * 70)
        print(f"\nTopic: {self.config.TOPIC_NAME}")
        print(f"Batch size: {batch_size} records")
        print(f"Interval: {interval_seconds} seconds")
        print("\nPress Ctrl+C to stop...")
        print("=" * 70)
        
        batch_number = 0
        
        try:
            while True:
                batch_number += 1
                start_time = time.time()
                
                # Generate synthetic data
                data = self.generate_batch(batch_size)
                
                # Send to Kafka
                sent = self.send_to_kafka(data, batch_number)
                
                elapsed = time.time() - start_time
                
                print(f"\n[Batch {batch_number}] Sent {sent} records to Kafka")
                print(f"Total sent: {self.total_sent}")
                print(f"Time: {elapsed:.2f}s")
                print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Sample data
                if batch_number == 1:
                    print("\nSample record:")
                    print(data.iloc[0].to_dict())
                
                # Wait for next batch
                sleep_time = max(0, interval_seconds - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("STREAM STOPPED BY USER")
            print("=" * 70)
            print(f"Total batches: {batch_number}")
            print(f"Total records sent: {self.total_sent}")
            print(f"Topic: {self.config.TOPIC_NAME}")
    
    
    def close(self):
        """Close Kafka producer"""
        if self.producer:
            self.producer.close()
            print("\nKafka producer closed")


def main():
    """Main entry point"""
    # Configuration
    DATA_PATH = "raw-data/Heart_disease_cleveland_new.csv"
    CATEGORICAL_COLUMNS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    
    # Streaming settings - Continuous mode only
    BATCH_SIZE = 10          # Records per batch
    INTERVAL_SECONDS = 2     # Seconds between batches
    
    # Initialize producer
    producer = KafkaStreamProducer()
    
    try:
        # Connect to Kafka
        if not producer.connect_kafka():
            return
        
        # Load and train model
        producer.load_and_train_model(DATA_PATH, CATEGORICAL_COLUMNS)
        
        # Start continuous streaming
        producer.stream_continuous(
            batch_size=BATCH_SIZE,
            interval_seconds=INTERVAL_SECONDS
        )
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        producer.close()


if __name__ == "__main__":
    main()

