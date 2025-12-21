"""
Kiểm tra model từ MinIO có đúng format không
"""
import boto3
import joblib
import tempfile
import os
import numpy as np

print("=" * 70)
print("CHECK MODEL FORMAT")
print("=" * 70)

# Connect to MinIO
s3_client = boto3.client(
    's3',
    endpoint_url='http://192.168.2.4:19000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin'
)

print("\n[1] Downloading model from MinIO...")
with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
    tmp_path = tmp.name

s3_client.download_file('ml-data', 'models/heart-disease/latest/model.joblib', tmp_path)
print(f"    Downloaded to: {tmp_path}")

print("\n[2] Loading model...")
model = joblib.load(tmp_path)
print(f"    Type: {type(model).__name__}")
print(f"    Model: {model}")

print("\n[3] Checking model attributes...")
print(f"    Has predict: {hasattr(model, 'predict')}")
print(f"    Has predict_proba: {hasattr(model, 'predict_proba')}")
print(f"    Has n_features_in_: {hasattr(model, 'n_features_in_')}")

if hasattr(model, 'n_features_in_'):
    print(f"    Expected features: {model.n_features_in_}")

print("\n[4] Testing prediction...")
# Sample với 15 features (standardized)
sample = np.array([[0.5, 1.0, 2.0, 0.3, -0.5, 0.0, 1.0, -0.8, 1.0, 1.2, 1.0, 0.0, 2.0, -0.4, 0.2]])
print(f"    Sample shape: {sample.shape}")

try:
    pred = model.predict(sample)
    print(f"    [OK] Prediction: {pred}")
    print(f"    Prediction type: {type(pred)}")
except Exception as e:
    print(f"    [FAIL] Error: {e}")

# Cleanup
os.unlink(tmp_path)

print("\n" + "=" * 70)

