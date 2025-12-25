import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime

class DualSinkStreaming:
    def __init__(self):
        # Kafka config
        self.kafka_bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'stream-project-kafka:9092')
        self.kafka_topic = os.getenv('KAFKA_TOPIC', 'heart-disease-stream')
        
        # Elasticsearch config
        self.es_nodes = os.getenv('ES_NODES', 'stream-project-elasticsearch')
        self.es_port = os.getenv('ES_PORT', '9200')
        self.es_index = os.getenv('ES_INDEX', 'heart-disease-stream')
        
        # MinIO config
        self.minio_endpoint = os.getenv('MINIO_ENDPOINT', 'http://stream-project-minio:9000')
        self.minio_access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.minio_secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        self.minio_bucket = os.getenv('MINIO_BUCKET', 'ml-data')
        
        self.checkpoint_location = "/app/spark/checkpoint"
        self.spark = None
    
    def create_spark_session(self):
        """Create Spark session with all required configurations"""
        print("\n" + "=" * 70)
        print("INITIALIZING SPARK SESSION (DUAL SINK: ES + MinIO)")
        print("=" * 70)
        
        print("\n[1] Configuring Spark session...")
        print(f"Kafka: {self.kafka_bootstrap}")
        print(f"Elasticsearch: {self.es_nodes}:{self.es_port}")
        print(f"MinIO: {self.minio_endpoint}")
        
        self.spark = SparkSession.builder \
            .appName("HeartDisease-DualSink-ES-MinIO") \
            .config("spark.jars.packages",
                    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.4,"
                    "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0,"
                    "org.apache.hadoop:hadoop-aws:3.3.4") \
            .config("spark.hadoop.fs.s3a.endpoint", self.minio_endpoint) \
            .config("spark.hadoop.fs.s3a.access.key", self.minio_access_key) \
            .config("spark.hadoop.fs.s3a.secret.key", self.minio_secret_key) \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
            .config("spark.sql.streaming.checkpointLocation", self.checkpoint_location) \
            .config("spark.es.nodes", self.es_nodes) \
            .config("spark.es.port", self.es_port) \
            .config("spark.es.nodes.wan.only", "true") \
            .config("spark.es.index.auto.create", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        print(f"\nSpark version: {self.spark.version}")
        print("Spark session created successfully!")
    
    def define_schema(self):
        """Define message schema from Kafka"""
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
            .option("failOnDataLoss", "false") \
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
        
        # Parse JSON from Kafka
        parsed_df = df.selectExpr("CAST(value AS STRING) as json_str") \
            .select(from_json(col("json_str"), message_schema).alias("data")) \
            .select("data.*")
        
        # Flatten and add metadata
        processed_df = parsed_df.select(
            col("batch_number"),
            col("record_id"),
            to_timestamp(col("timestamp")).alias("event_timestamp"),
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
        ) \
        .withColumn("processed_at", current_timestamp()) \
        .withColumn("date", to_date(col("event_timestamp"))) \
        .withColumn("hour", hour(col("event_timestamp")))
        
        print("Data processing pipeline configured!")
        print("\nOutput schema:")
        processed_df.printSchema()
        
        return processed_df
    
    def write_dual_sink(self, df):
        """Write to BOTH Elasticsearch AND MinIO using foreachBatch"""
        print("\n" + "=" * 70)
        print("CONFIGURING DUAL SINK (ES + MinIO)")
        print("=" * 70)
        
        def foreach_batch_function(batch_df, batch_id):
            """Process each micro-batch - write to both sinks"""
            
            record_count = batch_df.count()
            
            if record_count == 0:
                print(f"\n[Batch {batch_id}] Empty batch, skipping...")
                return
            
            print(f"\n{'=' * 70}")
            print(f"PROCESSING BATCH {batch_id}")
            print(f"{'=' * 70}")
            print(f"Records: {record_count}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show sample record
            print("\nSample record:")
            batch_df.select("record_id", "age", "sex", "target").show(1, truncate=False)
            
            # ================================
            # SINK 1: Elasticsearch (Real-time query)
            # ================================
            print(f"\n[Batch {batch_id}] Writing to Elasticsearch...")
            try:
                batch_df.write \
                    .format("org.elasticsearch.spark.sql") \
                    .option("es.resource", self.es_index) \
                    .option("es.nodes", self.es_nodes) \
                    .option("es.port", self.es_port) \
                    .option("es.nodes.wan.only", "true") \
                    .mode("append") \
                    .save()
                print(f"Elasticsearch: {record_count} records written")
            except Exception as e:
                print(f"Elasticsearch write failed: {e}")
            
            # ================================
            # SINK 2: MinIO (ML training data)
            # ================================
            print(f"\n[Batch {batch_id}] Writing to MinIO (Parquet)...")
            try:
                # Output path with partitioning
                output_path = f"s3a://{self.minio_bucket}/raw/heart-disease"
                
                # Write as Parquet with partitioning by date and hour
                batch_df \
                    .repartition(1) \
                    .write \
                    .format("parquet") \
                    .partitionBy("date", "hour") \
                    .mode("append") \
                    .option("compression", "snappy") \
                    .save(output_path)
                
                print(f" MinIO: {record_count} records written")
                print(f"   Path: {output_path}/date=.../hour=.../")
                print(f"   Format: Parquet (Snappy compressed)")
            except Exception as e:
                print(f" MinIO write failed: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n[Batch {batch_id}]  Completed successfully!")
            print("=" * 70)
        
        # Start streaming with foreachBatch
        query = df.writeStream \
            .foreachBatch(foreach_batch_function) \
            .trigger(processingTime="5 seconds") \
            .option("checkpointLocation", self.checkpoint_location) \
            .start()
        
        print("\n" + "=" * 70)
        print("DUAL SINK STREAMING ACTIVE")
        print("=" * 70)
        print(f"\n Sink 1: Elasticsearch")
        print(f"   └─ Index: {self.es_index}")
        print(f"   └─ Endpoint: http://localhost:19200/{self.es_index}")
        print(f"\nSink 2: MinIO (Parquet)")
        print(f"   └─ Bucket: {self.minio_bucket}")
        print(f"   └─ Path: s3a://{self.minio_bucket}/raw/heart-disease/")
        print(f"   └─ Console: http://localhost:19001")
        print("\n" + "=" * 70)
        print("Trigger: Every 5 seconds")
        print("Press Ctrl+C to stop...")
        print("=" * 70)
        
        return query
    
    def run(self):
        """Main execution"""
        try:
            self.create_spark_session()
            kafka_df = self.read_from_kafka()
            processed_df = self.process_data(kafka_df)
            query = self.write_dual_sink(processed_df)
            query.awaitTermination()
        except KeyboardInterrupt:
            print("\n\nStopping streaming...")
            if self.spark:
                self.spark.stop()
            print("Spark session stopped.")
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()
            if self.spark:
                self.spark.stop()

if __name__ == "__main__":
    app = DualSinkStreaming()
    app.run()

