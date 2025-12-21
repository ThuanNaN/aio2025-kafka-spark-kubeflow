"""
Test Elasticsearch queries
"""

import requests
import json
from es_config import ElasticsearchConfig

class ElasticsearchTester:
    def __init__(self):
        self.config = ElasticsearchConfig()
        self.base_url = f"http://{self.config.ES_HOSTS[0]}:{self.config.ES_PORT}"
        
    def test_connection(self):
        """Test ES connection"""
        print("=" * 70)
        print("ELASTICSEARCH CONNECTION TEST")
        print("=" * 70)
        
        try:
            response = requests.get(self.base_url)
            if response.status_code == 200:
                info = response.json()
                print(f"\nConnected to Elasticsearch!")
                print(f"Version: {info['version']['number']}")
                print(f"Cluster: {info['cluster_name']}")
                return True
            else:
                print(f"\nFailed: {response.status_code}")
                return False
        except Exception as e:
            print(f"\nConnection error: {e}")
            return False
    
    def get_index_info(self):
        """Get index information"""
        print("\n" + "=" * 70)
        print("INDEX INFORMATION")
        print("=" * 70)
        
        url = f"{self.base_url}/{self.config.INDEX_NAME}"
        response = requests.get(url)
        
        if response.status_code == 200:
            info = response.json()
            index_info = info[self.config.INDEX_NAME]
            
            print(f"\nIndex: {self.config.INDEX_NAME}")
            print(f"Status: EXISTS")
            print(f"Shards: {index_info['settings']['index']['number_of_shards']}")
            print(f"Replicas: {index_info['settings']['index']['number_of_replicas']}")
            print(f"Fields: {len(index_info['mappings']['properties'])}")
            return True
        else:
            print(f"\nIndex '{self.config.INDEX_NAME}' does not exist")
            print("Run: python elasticsearch/create_index.py")
            return False
    
    def count_documents(self):
        """Count total documents"""
        print("\n" + "=" * 70)
        print("DOCUMENT COUNT")
        print("=" * 70)
        
        url = f"{self.base_url}/{self.config.INDEX_NAME}/_count"
        response = requests.get(url)
        
        if response.status_code == 200:
            count = response.json()['count']
            print(f"\nTotal documents: {count}")
            return count
        else:
            print(f"\nError: {response.status_code}")
            return 0
    
    def get_latest_documents(self, size=5):
        """Get latest documents"""
        print("\n" + "=" * 70)
        print(f"LATEST {size} DOCUMENTS")
        print("=" * 70)
        
        query = {
            "size": size,
            "sort": [
                {"timestamp": {"order": "desc"}}
            ]
        }
        
        url = f"{self.base_url}/{self.config.INDEX_NAME}/_search"
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(query)
        )
        
        if response.status_code == 200:
            results = response.json()
            hits = results['hits']['hits']
            
            print(f"\nFound {len(hits)} documents:\n")
            
            for i, hit in enumerate(hits, 1):
                source = hit['_source']
                print(f"{i}. Record ID: {source.get('record_id')}")
                print(f"   Timestamp: {source.get('timestamp')}")
                print(f"   Age: {source.get('age')}, Sex: {source.get('sex')}, Target: {source.get('target')}")
                print()
            
            return hits
        else:
            print(f"\nError: {response.status_code}")
            return []
    
    def search_by_target(self, target_value):
        """Search documents by target value"""
        print("\n" + "=" * 70)
        print(f"SEARCH: TARGET = {target_value}")
        print("=" * 70)
        
        query = {
            "query": {
                "term": {
                    "target": target_value
                }
            },
            "size": 0,
            "aggs": {
                "count": {
                    "value_count": {
                        "field": "target"
                    }
                }
            }
        }
        
        url = f"{self.base_url}/{self.config.INDEX_NAME}/_search"
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(query)
        )
        
        if response.status_code == 200:
            results = response.json()
            count = results['hits']['total']['value']
            print(f"\nDocuments with target={target_value}: {count}")
            return count
        else:
            print(f"\nError: {response.status_code}")
            return 0
    
    def get_aggregations(self):
        """Get aggregations"""
        print("\n" + "=" * 70)
        print("AGGREGATIONS")
        print("=" * 70)
        
        query = {
            "size": 0,
            "aggs": {
                "by_target": {
                    "terms": {
                        "field": "target"
                    }
                },
                "by_sex": {
                    "terms": {
                        "field": "sex"
                    }
                },
                "avg_age": {
                    "avg": {
                        "field": "age"
                    }
                }
            }
        }
        
        url = f"{self.base_url}/{self.config.INDEX_NAME}/_search"
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(query)
        )
        
        if response.status_code == 200:
            results = response.json()
            aggs = results['aggregations']
            
            print("\nTarget distribution:")
            for bucket in aggs['by_target']['buckets']:
                print(f"  Target {bucket['key']}: {bucket['doc_count']} documents")
            
            print("\nSex distribution:")
            for bucket in aggs['by_sex']['buckets']:
                sex_label = "Male" if bucket['key'] == 1 else "Female"
                print(f"  {sex_label}: {bucket['doc_count']} documents")
            
            print(f"\nAverage age: {aggs['avg_age']['value']:.2f}")
            
            return aggs
        else:
            print(f"\nError: {response.status_code}")
            return None
    
    def run_all_tests(self):
        """Run all tests"""
        if not self.test_connection():
            return
        
        if not self.get_index_info():
            return
        
        count = self.count_documents()
        
        if count > 0:
            self.get_latest_documents(5)
            self.search_by_target(0)
            self.search_by_target(1)
            self.get_aggregations()
        else:
            print("\nNo documents in index yet.")
            print("Start streaming to populate data:")
            print("  python spark/spark_streaming_consumer.py")


def main():
    tester = ElasticsearchTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()

