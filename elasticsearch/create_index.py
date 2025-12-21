"""
Create Elasticsearch Index with proper mapping
Run this before starting Spark streaming
"""

import requests
import json
from es_config import ElasticsearchConfig

def create_index():
    """Create Elasticsearch index with mapping"""
    config = ElasticsearchConfig()
    
    url = f"http://{config.ES_HOSTS[0]}:{config.ES_PORT}/{config.INDEX_NAME}"
    
    print("=" * 70)
    print("CREATING ELASTICSEARCH INDEX")
    print("=" * 70)
    
    # Check if index exists
    print(f"\n[1] Checking if index '{config.INDEX_NAME}' exists...")
    response = requests.get(url)
    
    if response.status_code == 200:
        print(f"Index '{config.INDEX_NAME}' already exists")
        
        # Ask to delete
        delete = input("Delete and recreate? (y/n): ")
        if delete.lower() == 'y':
            print(f"Deleting index '{config.INDEX_NAME}'...")
            requests.delete(url)
            print("Deleted!")
        else:
            print("Keeping existing index")
            return
    
    # Create index
    print(f"\n[2] Creating index '{config.INDEX_NAME}'...")
    
    index_body = {
        "settings": config.INDEX_SETTINGS,
        "mappings": config.INDEX_MAPPING
    }
    
    response = requests.put(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(index_body)
    )
    
    if response.status_code in [200, 201]:
        print("Index created successfully!")
        print("\nIndex configuration:")
        print(json.dumps(index_body, indent=2))
    else:
        print(f"Error creating index: {response.status_code}")
        print(response.text)
        return
    
    # Verify index
    print(f"\n[3] Verifying index...")
    response = requests.get(url)
    
    if response.status_code == 200:
        print("Index verified successfully!")
        index_info = response.json()
        print(f"\nNumber of shards: {index_info[config.INDEX_NAME]['settings']['index']['number_of_shards']}")
        print(f"Number of replicas: {index_info[config.INDEX_NAME]['settings']['index']['number_of_replicas']}")
        print(f"Number of fields: {len(index_info[config.INDEX_NAME]['mappings']['properties'])}")
    
    print("\n" + "=" * 70)
    print("INDEX READY FOR STREAMING")
    print("=" * 70)
    print(f"\nIndex name: {config.INDEX_NAME}")
    print(f"Endpoint: http://{config.ES_HOSTS[0]}:{config.ES_PORT}")
    print("\nYou can now start Spark Streaming job")


def test_connection():
    """Test Elasticsearch connection"""
    config = ElasticsearchConfig()
    
    print("Testing Elasticsearch connection...")
    url = f"http://{config.ES_HOSTS[0]}:{config.ES_PORT}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            info = response.json()
            print(f"Connected to Elasticsearch {info['version']['number']}")
            return True
        else:
            print(f"Failed to connect: {response.status_code}")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        print("\nMake sure Elasticsearch is running:")
        print("  docker-compose up -d elasticsearch")
        return False


if __name__ == "__main__":
    if test_connection():
        create_index()

