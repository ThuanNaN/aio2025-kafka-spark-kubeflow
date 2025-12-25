"""
Kafka Configuration
"""

class KafkaConfig:
    # Kafka broker settings
    BOOTSTRAP_SERVERS = ['localhost:19092']
    
    # Topic settings
    TOPIC_NAME = 'heart-disease-stream'
    NUM_PARTITIONS = 1
    REPLICATION_FACTOR = 1
    
    # Producer settings
    PRODUCER_CONFIG = {
        'bootstrap_servers': BOOTSTRAP_SERVERS,
        'value_serializer': lambda v: v.encode('utf-8'),  # JSON string serialization
        'key_serializer': lambda k: k.encode('utf-8') if k else None,
        'acks': 'all',  # Wait for all replicas to acknowledge
        'retries': 3,
        'max_in_flight_requests_per_connection': 1,
        'compression_type': 'gzip',
        'api_version': (2, 8, 0),  # Compatible with Kafka 2.8+
    }
    
    # # Consumer settings
    # CONSUMER_CONFIG = {
    #     'bootstrap_servers': BOOTSTRAP_SERVERS,
    #     'auto_offset_reset': 'earliest',  # Start from beginning if no offset
    #     'enable_auto_commit': True,
    #     'group_id': 'heart-disease-consumer-group',
    #     'value_deserializer': lambda v: v.decode('utf-8'),
    #     'key_deserializer': lambda k: k.decode('utf-8') if k else None,
    #     'api_version': (2, 8, 0),  # Compatible with Kafka 2.8+
    # }
    
    # # Connection timeout
    # CONNECTION_TIMEOUT_MS = 10000
    # REQUEST_TIMEOUT_MS = 30000

