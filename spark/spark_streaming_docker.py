"""
Spark Structured Streaming Consumer for Docker
Reads from Kafka, processes, and writes to Elasticsearch
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, LongType

class SparkStreamingDocker:
    def __init__(self):
        # Get config from environment variables
        self.kafka_bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'stream-project-kafka:9092')
        self.kafka_topic = os.getenv('KAFKA_TOPIC', 'heart-disease-stream')
        self.es_nodes = os.getenv('ES_NODES', 'stream-project-elasticsearch')
        self.es_port = os.getenv('ES_PORT', '9200')
        self.es_index = os.getenv('ES_INDEX', 'heart-disease-stream')
        self.checkpoint_location = "/app/spark/checkpoint"
        self.spark = None
        
    def create_spark_session(self):
        """Create Spark session with required packages"""
        print("=" * 70)
        print("INITIALIZING SPARK SESSION (DOCKER)")
        print("=" * 70)
        
        print(f"\n[1] Creating Spark session...")
        print(f"Kafka: {self.kafka_bootstrap}")
        print(f"Elasticsearch: {self.es_nodes}:{self.es_port}")
        
        self.spark = SparkSession.builder \
            .appName("HeartDiseaseStreamingToES-Docker") \
            .config("spark.jars.packages", 
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.4,"
                    "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0") \
            .config("spark.sql.streaming.checkpointLocation", self.checkpoint_location) \
            .config("spark.es.nodes", self.es_nodes) \
            .config("spark.es.port", self.es_port) \
            .config("spark.es.nodes.wan.only", "true") \
            .config("spark.es.index.auto.create", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        print(f"Spark version: {self.spark.version}")
        print("Spark session created successfully!")
        
    def define_schema(self):
        """Define schema for incoming Kafka messages"""
        data_schema = StructType([
            StructField("age", FloatType(), True),
            StructField("sex", FloatType(), True),
            StructField("cp", FloatType(), True),
            StructField("trestbps", FloatType(), True),
            StructField("chol", FloatType(), True),
            StructField("fbs", FloatType(), True),
            StructField("restecg", FloatType(), True),
            StructField("thalach", FloatType(), True),
            StructField("exang", FloatType(), True),
            StructField("oldpeak", FloatType(), True),
            StructField("slope", FloatType(), True),
            StructField("ca", FloatType(), True),
            StructField("thal", FloatType(), True),
            StructField("target", FloatType(), True)
        ])
        
        message_schema = StructType([
            StructField("batch_number", IntegerType(), True),
            StructField("record_id", LongType(), True),
            StructField("timestamp", StringType(), True),
            StructField("data", data_schema, True)
        ])
        
        return message_schema
    
    def read_from_kafka(self):
        """Read streaming data from Kafka"""
        print("\n" + "=" * 70)
        print("READING FROM KAFKA")
        print("=" * 70)
        
        print(f"\n[2] Connecting to Kafka...")
        print(f"Bootstrap servers: {self.kafka_bootstrap}")
        print(f"Topic: {self.kafka_topic}")
        
        df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap) \
            .option("subscribe", self.kafka_topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        print("Connected to Kafka successfully!")
        return df
    
    def process_data(self, df):
        """Parse and transform data"""
        print("\n" + "=" * 70)
        print("PROCESSING DATA")
        print("=" * 70)
        
        print("\n[3] Parsing JSON messages...")
        
        message_schema = self.define_schema()
        
        parsed_df = df.selectExpr("CAST(value AS STRING) as json_str") \
            .select(from_json(col("json_str"), message_schema).alias("data")) \
            .select("data.*")
        
        flattened_df = parsed_df.select(
            col("batch_number"),
            col("record_id"),
            to_timestamp(col("timestamp")).alias("timestamp"),
            col("data.age").cast("integer").alias("age"),
            col("data.sex").cast("integer").alias("sex"),
            col("data.cp").cast("integer").alias("cp"),
            col("data.trestbps").cast("integer").alias("trestbps"),
            col("data.chol").cast("integer").alias("chol"),
            col("data.fbs").cast("integer").alias("fbs"),
            col("data.restecg").cast("integer").alias("restecg"),
            col("data.thalach").cast("integer").alias("thalach"),
            col("data.exang").cast("integer").alias("exang"),
            col("data.oldpeak").alias("oldpeak"),
            col("data.slope").cast("integer").alias("slope"),
            col("data.ca").cast("integer").alias("ca"),
            col("data.thal").cast("integer").alias("thal"),
            col("data.target").cast("integer").alias("target")
        )
        
        processed_df = flattened_df.withColumn("processed_at", current_timestamp())
        
        print("Data processing pipeline configured!")
        print("\nOutput schema:")
        processed_df.printSchema()
        
        return processed_df
    
    def write_to_elasticsearch(self, df):
        """Write streaming data to Elasticsearch"""
        print("\n" + "=" * 70)
        print("WRITING TO ELASTICSEARCH")
        print("=" * 70)
        
        print(f"\n[4] Configuring Elasticsearch sink...")
        print(f"ES nodes: {self.es_nodes}")
        print(f"ES port: {self.es_port}")
        print(f"ES index: {self.es_index}")
        
        query = df.writeStream \
            .format("es") \
            .outputMode("append") \
            .option("es.resource", self.es_index) \
            .option("es.nodes", self.es_nodes) \
            .option("es.port", self.es_port) \
            .option("es.nodes.wan.only", "true") \
            .option("checkpointLocation", self.checkpoint_location) \
            .trigger(processingTime="5 seconds") \
            .start()
        
        print("Streaming to Elasticsearch started!")
        return query
    
    def run(self):
        """Main execution method"""
        try:
            self.create_spark_session()
            kafka_df = self.read_from_kafka()
            processed_df = self.process_data(kafka_df)
            query = self.write_to_elasticsearch(processed_df)
            
            print("\n" + "=" * 70)
            print("STREAMING ACTIVE")
            print("=" * 70)
            print("\nMonitor:")
            print(f"  - Kafka UI: http://localhost:18080")
            print(f"  - Elasticsearch: http://localhost:19200/{self.es_index}/_search")
            print("=" * 70)
            
            query.awaitTermination()
            
        except KeyboardInterrupt:
            print("\n\nStopping streaming...")
            if self.spark:
                self.spark.stop()
            print("Streaming stopped!")
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            if self.spark:
                self.spark.stop()


if __name__ == "__main__":
    consumer = SparkStreamingDocker()
    consumer.run()

