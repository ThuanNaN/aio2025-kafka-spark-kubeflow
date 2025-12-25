"""
Elasticsearch Configuration
"""

class ElasticsearchConfig:
    # Elasticsearch connection
    ES_HOSTS = ["localhost"]
    ES_PORT = "19200"
    ES_HTTP_AUTH = None  # No authentication
    
    # Index settings
    INDEX_NAME = "heart-disease-stream"
    
    # Index settings
    INDEX_SETTINGS = {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "1s"  # Near real-time
    }
    
    # Field mappings
    INDEX_MAPPING = {
        "properties": {
            "batch_number": {"type": "integer"},
            "record_id": {"type": "long"},
            "timestamp": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis"
            },
            "age": {"type": "integer"},
            "sex": {"type": "keyword"},
            "cp": {"type": "keyword"},
            "trestbps": {"type": "integer"},
            "chol": {"type": "integer"},
            "fbs": {"type": "keyword"},
            "restecg": {"type": "keyword"},
            "thalach": {"type": "integer"},
            "exang": {"type": "keyword"},
            "oldpeak": {"type": "float"},
            "slope": {"type": "keyword"},
            "ca": {"type": "keyword"},
            "thal": {"type": "keyword"},
            "target": {"type": "keyword"}
        }
    }
