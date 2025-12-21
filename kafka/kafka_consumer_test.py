"""
Kafka Consumer for Testing
Consumes messages from heart-disease-stream topic and displays them
"""

import json
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from kafka_config import KafkaConfig
from datetime import datetime
import signal
import sys

class KafkaStreamConsumer:
    def __init__(self):
        self.config = KafkaConfig()
        self.consumer = None
        self.total_consumed = 0
        self.running = True
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\nShutting down consumer...")
        self.running = False
    
    def connect(self):
        """Connect to Kafka topic"""
        print("=" * 70)
        print("KAFKA CONSUMER - TESTING")
        print("=" * 70)
        
        try:
            print(f"\nConnecting to Kafka at {self.config.BOOTSTRAP_SERVERS}...")
            print(f"Topic: {self.config.TOPIC_NAME}")
            print(f"Group ID: {self.config.CONSUMER_CONFIG['group_id']}")
            
            self.consumer = KafkaConsumer(
                self.config.TOPIC_NAME,
                **self.config.CONSUMER_CONFIG
            )
            
            print("Successfully connected to Kafka!")
            print("\nWaiting for messages... (Press Ctrl+C to stop)")
            print("=" * 70)
            return True
            
        except Exception as e:
            print(f"\nError connecting to Kafka: {e}")
            print("\nMake sure:")
            print("  1. Kafka is running: docker-compose up -d")
            print("  2. Topic exists and producer is sending data")
            return False
    
    def consume_messages(self, max_messages=None, display_mode="summary"):
        """
        Consume messages from topic
        
        Args:
            max_messages: Maximum number of messages to consume (None = unlimited)
            display_mode: "full", "summary", or "compact"
        """
        if not self.consumer:
            print("Consumer not connected!")
            return
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                self.total_consumed += 1
                
                # Parse message
                try:
                    value = json.loads(message.value)
                    
                    # Display based on mode
                    if display_mode == "full":
                        self._display_full(message, value)
                    elif display_mode == "summary":
                        self._display_summary(message, value)
                    elif display_mode == "compact":
                        self._display_compact(message, value)
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding message: {e}")
                    print(f"Raw value: {message.value}")
                
                # Check if reached max messages
                if max_messages and self.total_consumed >= max_messages:
                    print(f"\nReached max messages limit: {max_messages}")
                    break
            
        except KafkaError as e:
            print(f"\nKafka error: {e}")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
        finally:
            self.close()
    
    def _display_full(self, message, value):
        """Display full message details"""
        print("\n" + "=" * 70)
        print(f"Message #{self.total_consumed}")
        print("-" * 70)
        print(f"Partition: {message.partition}")
        print(f"Offset: {message.offset}")
        print(f"Key: {message.key}")
        print(f"Timestamp: {datetime.fromtimestamp(message.timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nMessage Content:")
        print(json.dumps(value, indent=2))
        print("=" * 70)
    
    def _display_summary(self, message, value):
        """Display summary of message"""
        if self.total_consumed % 10 == 1:  # Header every 10 messages
            print(f"\n{'Count':<8} {'Batch':<8} {'Record ID':<12} {'Timestamp':<20} {'Target':<8}")
            print("-" * 70)
        
        data = value.get('data', {})
        print(f"{self.total_consumed:<8} "
              f"{value.get('batch_number', 'N/A'):<8} "
              f"{value.get('record_id', 'N/A'):<12} "
              f"{value.get('timestamp', 'N/A')[:19]:<20} "
              f"{data.get('target', 'N/A'):<8}")
    
    def _display_compact(self, message, value):
        """Display compact one-line format"""
        data = value.get('data', {})
        print(f"[{self.total_consumed}] Batch:{value.get('batch_number')} "
              f"ID:{value.get('record_id')} "
              f"Age:{data.get('age')} "
              f"Sex:{data.get('sex')} "
              f"Target:{data.get('target')}")
    
    def close(self):
        """Close consumer"""
        if self.consumer:
            self.consumer.close()
            print("\n" + "=" * 70)
            print("CONSUMER STOPPED")
            print("=" * 70)
            print(f"Total messages consumed: {self.total_consumed}")
            print("=" * 70)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Consumer for heart-disease-stream')
    parser.add_argument('--mode', choices=['full', 'summary', 'compact'], 
                       default='summary', help='Display mode')
    parser.add_argument('--max', type=int, default=None, 
                       help='Maximum number of messages to consume')
    
    args = parser.parse_args()
    
    # Create and run consumer
    consumer = KafkaStreamConsumer()
    
    if consumer.connect():
        consumer.consume_messages(
            max_messages=args.max,
            display_mode=args.mode
        )


if __name__ == "__main__":
    main()

