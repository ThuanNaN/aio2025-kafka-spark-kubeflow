#!/bin/bash

# Set Spark Home
export SPARK_HOME=${SPARK_HOME:-/opt/spark}

# Define Spark packages
SPARK_PACKAGES="org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.4,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0,org.apache.hadoop:hadoop-aws:3.3.4"

echo "======================================================================"
echo "Starting Spark Streaming - Dual Sink (ES + MinIO)"
echo "======================================================================"
echo "SPARK_HOME: $SPARK_HOME"
echo "Packages: Kafka, Elasticsearch, Hadoop-AWS (S3)"
echo ""

# Execute the Spark job
exec "${SPARK_HOME}/bin/spark-submit" \
  --packages "${SPARK_PACKAGES}" \
  --master local[*] \
  /app/spark/spark_streaming_dual_sink.py

# docker exec stream-project-spark /bin/bash /app/spark/run_dual_sink.sh