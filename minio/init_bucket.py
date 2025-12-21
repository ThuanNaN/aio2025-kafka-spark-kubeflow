import os
import sys

def init_minio_bucket():
    """
    Initialize MinIO bucket for ML data storage
    Uses MinIO client (mc) inside the MinIO container
    """
    bucket_name = "ml-data"
    
    print("=" * 70)
    print("INITIALIZING MINIO BUCKET")
    print("=" * 70)
    
    print(f"\n[1] Bucket name: {bucket_name}")
    print(f"[2] MinIO Console: http://localhost:19001")
    print(f"[3] Credentials: minioadmin / minioadmin")
    
    print("\n" + "=" * 70)
    print("MANUAL SETUP (via MinIO Console):")
    print("=" * 70)
    print("""
1. Open browser: http://localhost:19001
2. Login with:
   - Username: minioadmin
   - Password: minioadmin
3. Click "Create Bucket" (+ button)
4. Enter bucket name: ml-data
5. Click "Create"

✅ Bucket created successfully!
    """)
    
    print("\n" + "=" * 70)
    print("ALTERNATIVE: Use MinIO Client (mc)")
    print("=" * 70)
    print("""
# Inside MinIO container:
docker exec stream-project-minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker exec stream-project-minio mc mb local/ml-data
docker exec stream-project-minio mc ls local/

✅ Done!
    """)
    
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print("""
After Spark writes data, you can check:

1. MinIO Console:
   http://localhost:19001 → Browse ml-data bucket

2. Expected structure:
   ml-data/
   └── raw/
       └── heart-disease/
           └── date=2025-12-21/
               └── hour=03/
                   └── part-00000.parquet
    """)

if __name__ == "__main__":
    init_minio_bucket()

