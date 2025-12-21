"""
Deploy Heart Disease Model to KServe
Can be used as:
1. Pipeline component (integrated into pipeline)
2. Standalone script (deploy existing model from MinIO)
"""

from kfp import dsl, compiler
from kfp.dsl import Artifact, Input, Output
from typing import NamedTuple


# ============================================================================
# HELPER: Create MinIO Secret
# ============================================================================
def create_minio_secret(
    namespace: str,
    minio_endpoint: str = "192.168.2.4:19000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin"
):
    """
    Create MinIO credentials secret for KServe storage-initializer
    """
    from kubernetes import client
    import base64
    
    print(f"\n[1] Creating MinIO credentials secret...")
    
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
            "AWS_ACCESS_KEY_ID": access_key,
            "AWS_SECRET_ACCESS_KEY": secret_key
        }
    }
    
    v1 = client.CoreV1Api()
    
    try:
        v1.create_namespaced_secret(namespace=namespace, body=secret)
        print(f"    [OK] Secret '{secret_name}' created")
    except client.exceptions.ApiException as e:
        if e.status == 409:
            print(f"    [OK] Secret '{secret_name}' already exists")
        else:
            raise
    
    return secret_name


# ============================================================================
# HELPER: Create ServiceAccount
# ============================================================================
def create_service_account(namespace: str, secret_name: str):
    """
    Create ServiceAccount with MinIO secret attached
    """
    from kubernetes import client
    
    print(f"\n[2] Creating ServiceAccount...")
    
    sa_name = "kserve-sa"
    
    service_account = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {
            "name": sa_name,
            "namespace": namespace
        },
        "secrets": [
            {"name": secret_name}
        ]
    }
    
    v1 = client.CoreV1Api()
    
    try:
        v1.create_namespaced_service_account(namespace=namespace, body=service_account)
        print(f"    [OK] ServiceAccount '{sa_name}' created")
    except client.exceptions.ApiException as e:
        if e.status == 409:
            print(f"    [OK] ServiceAccount '{sa_name}' already exists")
        else:
            raise
    
    return sa_name


# ============================================================================
# STANDALONE DEPLOYMENT SCRIPT
# ============================================================================
def deploy_standalone(
    model_uri: str = "s3://ml-data/models/heart-disease/latest/model.joblib",
    model_version: str = "latest",
    service_name: str = "heart-disease-predictor",
    namespace: str = "kubeflow-user-example-com",
    minio_endpoint: str = "192.168.2.4:19000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin"
):
    """
    Standalone function to deploy model directly (not in pipeline)
    
    Usage:
        python demo_kserve.py deploy
    """
    from kubernetes import client, config
    import yaml
    
    print("=" * 70)
    print("STANDALONE KSERVE DEPLOYMENT")
    print("=" * 70)
    
    # Load config
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()
    
    # Step 1: Create MinIO secret
    secret_name = create_minio_secret(
        namespace=namespace,
        minio_endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key
    )
    
    # Step 2: Create ServiceAccount
    sa_name = create_service_account(namespace=namespace, secret_name=secret_name)
    
    # Step 3: Create InferenceService
    print(f"\n[3] Creating InferenceService...")
    
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": service_name,
            "namespace": namespace
        },
        "spec": {
            "predictor": {
                "serviceAccountName": sa_name,  # USE THE SERVICE ACCOUNT
                "sklearn": {
                    "storageUri": model_uri,
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {"cpu": "1", "memory": "1Gi"}
                    }
                }
            }
        }
    }
    
    print(f"    Model URI: {model_uri}")
    print(f"    Service: {service_name}")
    print(f"    Namespace: {namespace}")
    print(f"    ServiceAccount: {sa_name}")
    
    api = client.CustomObjectsApi()
    
    try:
        # Try to create
        api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            body=inference_service
        )
        print(f"    [OK] InferenceService created!")
        
    except client.exceptions.ApiException as e:
        if e.status == 409:  # Already exists
            print(f"    [WARN] InferenceService already exists - updating...")
            api.patch_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=service_name,
                body=inference_service
            )
            print(f"    [OK] InferenceService updated!")
        else:
            raise
    
    print("\n" + "=" * 70)
    print("[OK] DEPLOYMENT COMPLETE!")
    print("=" * 70)
    
    print(f"\n[CHECK STATUS]")
    print(f"  kubectl get inferenceservice {service_name} -n {namespace}")
    print(f"  kubectl describe inferenceservice {service_name} -n {namespace}")
    
    print(f"\n[GET LOGS]")
    print(f"  kubectl logs -n {namespace} -l serving.kserve.io/inferenceservice={service_name} --all-containers --tail=50")
    
    print(f"\n[GET URL]")
    print(f"  kubectl get inferenceservice {service_name} -n {namespace} -o jsonpath='{{.status.url}}'")
    
    print(f"\n[TEST PREDICTION]")
    print(f"""  # Get URL
  URL=$(kubectl get inferenceservice {service_name} -n {namespace} -o jsonpath='{{.status.url}}')
  
  # Send prediction request
  curl -X POST $URL/v1/models/{service_name}:predict \\
    -H 'Content-Type: application/json' \\
    -d '{{
      "instances": [
        [0.5, -0.2, 1.1, 0.3, -0.7, 0.9, -0.4, 1.2, 0.6, -0.5, 
         0.8, -0.3, 0.4, 0.7, 0.2, 0.1]
      ]
    }}'
    """)
    
    print("\n" + "=" * 70)


# ============================================================================
# HELPER: Create inference request
# ============================================================================
def create_sample_inference_request():
    """
    Create a sample prediction request for the heart disease model
    """
    import json
    
    # Sample feature values (standardized)
    # These correspond to: age, sex, cp, trestbps, chol, fbs, restecg, 
    # thalach, exang, oldpeak, slope, ca, thal, age_x_thalach, bmi_proxy
    sample_features = [
        0.5,   # age (standardized)
        1.0,   # sex
        2.0,   # cp
        0.3,   # trestbps
        -0.5,  # chol
        0.0,   # fbs
        1.0,   # restecg
        -0.8,  # thalach
        1.0,   # exang
        1.2,   # oldpeak
        1.0,   # slope
        0.0,   # ca
        2.0,   # thal
        -0.4,  # age_x_thalach (derived)
        0.2    # bmi_proxy (derived)
    ]
    
    request = {
        "instances": [sample_features]
    }
    
    print("Sample Prediction Request:")
    print(json.dumps(request, indent=2))
    
    return request


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("KSERVE DEPLOYMENT HELPER")
    print("=" * 70)
    
    if len(sys.argv) > 1 and sys.argv[1] == "deploy":
        # Deploy standalone
        deploy_standalone()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "sample":
        # Create sample request
        create_sample_inference_request()
    
    else:
        print("\nUsage:")
        print("  python demo_kserve.py deploy       # Deploy model to KServe")
        print("  python demo_kserve.py sample       # Show sample prediction request")
        print("\nOr import as KFP component:")
        print("  from demo_kserve import deploy_model_to_kserve")