@echo off
echo ======================================================================
echo Running Spark Streaming with Official Spark Image
echo ======================================================================

echo.
echo [1] Checking if Kafka Producer is running...
docker ps | findstr kafka-producer >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Kafka Producer may not be running!
    echo Please start producer first: python kafka/kafka_producer.py
    echo.
)

echo [2] Starting Spark Structured Streaming job...
echo.

docker exec -it stream-project-spark /bin/bash /app/spark/run_streaming.sh

echo.
echo ======================================================================
echo Spark Streaming job ended
echo ======================================================================
pause
