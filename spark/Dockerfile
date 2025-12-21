FROM spark:3.4.4-python3

USER root

# Install Python dependencies (including PySpark for Python API)
RUN pip install --no-cache-dir pyspark==3.4.4 elasticsearch requests

# Create app directory
WORKDIR /app

# Copy Spark scripts
COPY spark/*.py /app/spark/
COPY spark/*.sh /app/spark/

# Make shell scripts executable
RUN chmod +x /app/spark/*.sh

# Create checkpoint directory
RUN mkdir -p /app/spark/checkpoint && chmod -R 777 /app/spark/checkpoint

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin
ENV PYTHONPATH=/app:$PYTHONPATH

# Keep container running
CMD ["tail", "-f", "/dev/null"]
#docker exec stream-project-spark /bin/bash /app/spark/run_streaming.sh
