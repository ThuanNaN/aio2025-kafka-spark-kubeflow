# Heart Disease ML Project
Real-time Streaming, Dual Sink (Elasticsearch + MinIO), Kubeflow Pipeline, and KServe.

![Project Architecture](https://res.cloudinary.com/dptjhpkmv/image/upload/v1766686516/stream_project_tmssic.png)

## Overview
- Real-time data generation with Kafka (SDV-based synthetic data)
- Spark Structured Streaming processes Kafka messages and writes to:
	- Elasticsearch for real-time search/analytics
	- MinIO (S3) as Parquet for ML pipelines
- Kubeflow pipeline compiles and runs training/evaluation and registers models to MinIO
- KServe deploys the latest model from MinIO for online inference

## Prerequisites
- Docker and Docker Compose
- Python 3.11+

## Setup
1) Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Start infrastructure (Kafka, Kafka UI, Elasticsearch, MinIO, Spark):

```bash
docker-compose up -d
```

Services and UIs:
- Kafka UI: http://localhost:18080
- Elasticsearch: http://localhost:19200
- MinIO API: http://localhost:19000
- MinIO Console: http://localhost:19001 (login: `minioadmin` / `minioadmin`)

## Elasticsearch Index (Dual Sink target)
Create the index and mapping used by Spark streaming:

```bash
python elasticsearch/create_index.py
```

Basic checks and queries:

```bash
python elasticsearch/test_es.py
```

## MinIO Bucket (Dual Sink target)
Create bucket `ml-data` in MinIO Console (http://localhost:19001) before streaming.
Spark writes Parquet to: `s3a://ml-data/raw/heart-disease/` partitioned by `date` and `hour`.

## Run Spark Streaming (Dual Sink: ES + MinIO)
Execute the streaming job inside the Spark container:

```bash
docker exec stream-project-spark /bin/bash /app/spark/run_dual_sink.sh
```

This job:
- Reads Kafka topic `heart-disease-stream`
- Writes processed records to Elasticsearch index `heart-disease-stream`
- Writes Parquet files to MinIO bucket `ml-data` under `raw/heart-disease/`

## Produce Streaming Data (Kafka)
The producer trains an SDV model using a CSV dataset, then streams synthetic records to Kafka.

1) Prepare dataset (required once):
- Place the CSV at `raw-data/Heart_disease_cleveland_new.csv` (or update the path in `kafka/kafka_producer.py`)
- Set categorical columns as defined in the script

2) Run the producer:

```bash
python kafka/kafka_producer.py
```

Defaults (can be adjusted in `kafka/kafka_config.py`):
- Bootstrap servers: `localhost:19092`
- Topic: `heart-disease-stream`
- Producer: gzip compression, acks=all

## Verify Data Flow
- Elasticsearch: run `python elasticsearch/test_es.py` for index info, counts, latest docs, aggregations
- MinIO: check Parquet files in `ml-data/raw/heart-disease/` via Console (Browse) or S3 client
- Kafka: use Kafka UI (http://localhost:18080) to inspect topic data

## Kubeflow Pipeline (Training + Registry)
Compile the pipeline to a YAML package for uploading to Kubeflow:

```bash
python kubeflow_pipeline/heart_disease_pipeline.py
```

Upload `heart_disease_pipeline_simple.yaml` to Kubeflow Pipelines UI and set parameters:
- `minio_endpoint`: for local Docker use `http://localhost:19000`
- `bucket_name`: `ml-data`
- `data_prefix`: `raw/heart-disease`

Pipeline steps:
1) Prepare data (collect from MinIO, preprocess, features)
2) Validate data (basic quality checks)
3) Train & Evaluate (RandomForest, fixed hyperparameters)
4) Register model (store model + metadata to MinIO)

## KServe Deployment
Deploy the latest registered model from MinIO to KServe:

```bash
python kubeflow_pipeline/demo_kserve.py deploy
```

Requirements:
- KServe installed in the cluster
- Namespace (default): `kubeflow-user-example-com`
- MinIO endpoint configured (defaults can be changed in the script)

## Notes
- Default index and topic: `heart-disease-stream`
- Spark checkpoint path: `/app/spark/checkpoint` (mounted volume)
- Parquet files are written with Snappy compression and partitioned by date/hour
