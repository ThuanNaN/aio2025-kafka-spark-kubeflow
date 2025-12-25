"""
Deploy Heart Disease Model to KServe
Standalone script to deploy existing model from MinIO to KServe
"""

from kubernetes import client, config
import boto3


def deploy_standalone(
    model_uri: str = None,
    service_name: str = "heart-disease-predictor",
    namespace: str = "kubeflow-user-example-com",
    minio_endpoint: str = "192.168.2.4:19000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
    bucket_name: str = "ml-data"
):
    """
    Deploy model to KServe
    
    Args:
        model_uri: Model path in S3. If None, automatically find latest model
        service_name: Name of InferenceService
        namespace: Kubernetes namespace
        minio_endpoint: MinIO endpoint
        minio_access_key: MinIO access key
        minio_secret_key: MinIO secret key
        bucket_name: MinIO bucket name
    """
    print("=" * 70)
    print("KSERVE DEPLOYMENT")
    print("=" * 70)
    
    # Load Kubernetes config
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()
    
    # Find latest model if not specified
    if model_uri is None:
        print("\n[Finding latest model in MinIO]")
        minio_endpoint_url = f"http://{minio_endpoint}" if not minio_endpoint.startswith("http") else minio_endpoint
        
        s3_client = boto3.client(
            's3',
            endpoint_url=minio_endpoint_url,
            aws_access_key_id=minio_access_key,
            aws_secret_access_key=minio_secret_key,
            verify=False
        )
        
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix="models/heart-disease/"
            )
            
            if 'Contents' not in response:
                print("[ERROR] No models found!")
                return
            
            model_paths = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('/model.joblib')]
            
            if not model_paths:
                print("[ERROR] No model.joblib files found!")
                return
            
            # Sort by modified time (newest first)
            model_paths_with_time = []
            for path in model_paths:
                obj_info = s3_client.head_object(Bucket=bucket_name, Key=path)
                model_paths_with_time.append({
                    'path': path,
                    'modified': obj_info['LastModified']
                })
            
            model_paths_with_time.sort(key=lambda x: x['modified'], reverse=True)
            latest_model = model_paths_with_time[0]
            model_uri = f"s3://{bucket_name}/{latest_model['path']}"
            print(f"[OK] Found latest model: {model_uri}")
            
        except Exception as e:
            print(f"[ERROR] Error finding model: {e}")
            return
    
    # Create MinIO secret
    print("\n[1] Creating MinIO secret...")
    secret_name = "minio-s3-secret"
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": secret_name,
            "namespace": namespace,
            "annotations": {
                "serving.kserve.io/s3-endpoint": minio_endpoint,
                "serving.kserve.io/s3-usehttps": "0",
                "serving.kserve.io/s3-region": "us-east-1",
                "serving.kserve.io/s3-useanoncredential": "false"
            }
        },
        "type": "Opaque",
        "stringData": {
            "AWS_ACCESS_KEY_ID": minio_access_key,
            "AWS_SECRET_ACCESS_KEY": minio_secret_key
        }
    }
    
    v1 = client.CoreV1Api()
    try:
        v1.create_namespaced_secret(namespace=namespace, body=secret)
        print(f"[OK] Secret '{secret_name}' created")
    except client.exceptions.ApiException as e:
        if e.status == 409:
            print(f"[OK] Secret '{secret_name}' already exists")
        else:
            raise
    
    # Create ServiceAccount
    print("\n[2] Creating ServiceAccount...")
    sa_name = "kserve-sa"
    service_account = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {
            "name": sa_name,
            "namespace": namespace
        },
        "secrets": [{"name": secret_name}]
    }
    
    try:
        v1.create_namespaced_service_account(namespace=namespace, body=service_account)
        print(f"[OK] ServiceAccount '{sa_name}' created")
    except client.exceptions.ApiException as e:
        if e.status == 409:
            print(f"[OK] ServiceAccount '{sa_name}' already exists")
        else:
            raise
    
    # Create InferenceService
    print("\n[3] Creating InferenceService...")
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": service_name,
            "namespace": namespace
        },
        "spec": {
            "predictor": {
                "serviceAccountName": sa_name,
                "model": {
                    "modelFormat": {"name": "sklearn"},
                    "storageUri": model_uri,
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {"cpu": "1", "memory": "1Gi"}
                    }
                }
            }
        }
    }
    
    api = client.CustomObjectsApi()
    try:
        api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            body=inference_service
        )
        print(f"[OK] InferenceService '{service_name}' created")
    except client.exceptions.ApiException as e:
        if e.status == 409:
            print(f"[WARN] InferenceService already exists - updating...")
            api.patch_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=service_name,
                body=inference_service
            )
            print(f"[OK] InferenceService updated")
        else:
            raise
    
    print("\n" + "=" * 70)
    print("[OK] DEPLOYMENT COMPLETE!")
    print("=" * 70)
    print(f"\nCheck status:")
    print(f"  kubectl get inferenceservice {service_name} -n {namespace}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        deploy_standalone()
    else:
        print("Usage:")
        print("  python demo_kserve.py deploy         # Deploy model to KServe (auto-find latest model)")
