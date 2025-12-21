#!/bin/bash
# Run Spark Structured Streaming job
# docker exec -it stream-project-spark /bin/bash /app/spark/run_streaming.sh
echo "======================================================================"
echo "Starting Spark Structured Streaming - Heart Disease to Elasticsearch"
echo "======================================================================"

/opt/spark/bin/spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.4,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0 \
  --master local[*] \
  /app/spark/spark_streaming_docker.py

