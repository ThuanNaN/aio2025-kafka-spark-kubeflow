"""
Kubeflow Pipeline: Heart Disease with Katib (DEBUG VERSION)
- Shorter timeout for faster debugging
- Better logging
- Simpler training function
"""

from kfp import dsl, compiler
from kfp.dsl import Artifact, Input, Output
from typing import NamedTuple


# ============================================================================
# COMPONENT 1: MERGED Data Preparation
# ============================================================================
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["boto3", "pandas", "pyarrow", "numpy", "scikit-learn"]
)
def prepare_data(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    data_prefix: str,
    output_data: Output[Artifact]
) -> NamedTuple('Outputs', [('num_records', int), ('num_features', int)]):
    """MERGED: Collect + Preprocess + Feature Engineering"""
    import boto3
    import pandas as pd
    import numpy as np
    import tempfile
    import os
    from sklearn.preprocessing import StandardScaler
    from collections import namedtuple
    
    print("=" * 70)
    print("COMPONENT 1: DATA PREPARATION")
    print("=" * 70)
    
    # Collect from MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        verify=False
    )
    
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=data_prefix)
    parquet_files = [obj['Key'] for obj in response.get('Contents', []) 
                    if obj['Key'].endswith('.parquet')]
    
    dfs = []
    for file_key in parquet_files[:10]:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
        tmp_file_path = tmp_file.name
        tmp_file.close()
        s3_client.download_file(bucket_name, file_key, tmp_file_path)
        dfs.append(pd.read_parquet(tmp_file_path))
        os.unlink(tmp_file_path)
    
    if not dfs:
        raise Exception("No data found in MinIO!")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Preprocess
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                   'ca', 'thal', 'target']
    df_clean = combined_df[feature_cols].copy()
    df_clean = df_clean.fillna(df_clean.median())
    df_clean = df_clean.drop_duplicates()
    
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_clean[col] = df_clean[col].clip(Q1 - 3*IQR, Q3 + 3*IQR)
    
    # Feature engineering
    df_clean['age_x_thalach'] = df_clean['age'] * df_clean['thalach']
    df_clean['bmi_proxy'] = df_clean['chol'] / (df_clean['thalach'] + 1)
    
    target = df_clean['target']
    features = df_clean.drop('target', axis=1)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_df = pd.DataFrame(features_scaled, columns=features.columns)
    
    df_final = features_df.copy()
    df_final['target'] = target.values
    
    df_final.to_csv(output_data.path, index=False)
    print(f"✅ Complete! {len(df_final)} records, {len(features.columns)} features")
    
    Outputs = namedtuple('Outputs', ['num_records', 'num_features'])
    return Outputs(len(df_final), len(features.columns))


# ============================================================================
# COMPONENT 2: Validate Data
# ============================================================================
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy"]
)
def validate_data(
    input_data: Input[Artifact],
    min_records: int = 100,
    min_features: int = 10
) -> str:
    """Validate data quality"""
    import pandas as pd
    import json
    
    print("COMPONENT 2: DATA VALIDATION")
    
    df = pd.read_csv(input_data.path)
    
    validation_checks = {
        'min_records': bool(len(df) >= min_records),
        'min_features': bool((len(df.columns) - 1) >= min_features),
        'no_missing': bool(df.isnull().sum().sum() == 0),
        'balanced_target': bool(df['target'].value_counts().min() >= 30)
    }
    
    is_valid = all(validation_checks.values())
    
    validation_report = json.dumps({
        'total_records': int(len(df)),
        'total_features': int(len(df.columns) - 1),
        'checks': validation_checks,
        'is_valid': bool(is_valid)
    }, indent=2)
    
    print(f"Result: {'✅ VALID' if is_valid else '❌ INVALID'}")
    
    if not is_valid:
        raise Exception(f"Validation failed!")
    
    return validation_report


# ============================================================================
# COMPONENT 3: Katib HPO (DEBUG VERSION with detailed logging)
# ============================================================================
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["kubeflow-katib==0.17.0", "pandas", "scikit-learn"]
)
def katib_hyperparameter_tuning_debug(
    input_data: Input[Artifact],
    katib_namespace: str = "kubeflow-user-example-com",
    max_trials: int = 3,  # Reduced for faster debugging
    parallel_trials: int = 1  # Run sequentially for easier debugging
) -> NamedTuple('Outputs', [('best_n_estimators', int), ('best_max_depth', int), ('best_accuracy', float)]):
    """
    Katib HPO with DEBUG mode - reduced trials and better logging
    """
    from kubeflow import katib
    import pandas as pd
    from collections import namedtuple
    import time
    
    print("=" * 70)
    print("COMPONENT 3: KATIB HPO (DEBUG MODE)")
    print("=" * 70)
    
    data_path = input_data.path
    print(f"\n[DEBUG] Data artifact path: {data_path}")
    
    # Define SIMPLE training function for debugging
    def train_heart_disease_simple(parameters):
        """
        SIMPLIFIED training function for debugging
        Just prints parameters and returns a simulated accuracy
        """
        import sys
        
        print("="*50)
        print("KATIB TRIAL STARTED")
        print("="*50)
        
        # Get hyperparameters
        n_estimators = int(parameters["n_estimators"])
        max_depth = int(parameters["max_depth"])
        
        print(f"Hyperparameters received:")
        print(f"  n_estimators = {n_estimators}")
        print(f"  max_depth = {max_depth}")
        
        # Simulate training (no actual data needed for debugging)
        print("\nSimulating training...")
        
        # Calculate simulated accuracy based on hyperparameters
        # Better params = higher accuracy
        base_acc = 0.75
        n_est_bonus = (n_estimators - 50) / 1500  # Max +0.10
        depth_bonus = (max_depth - 5) / 150       # Max +0.10
        
        accuracy = base_acc + n_est_bonus + depth_bonus
        accuracy = min(accuracy, 0.95)
        
        print(f"\nSimulated training complete!")
        print(f"Simulated accuracy: {accuracy:.4f}")
        
        # CRITICAL: Print metric in exact format Katib expects
        print(f"\naccuracy={accuracy:.4f}")
        
        print("="*50)
        print("KATIB TRIAL FINISHED")
        print("="*50)
        
        sys.stdout.flush()  # Force flush output
        
        return accuracy
    
    # Define search space
    print("\n[DEBUG] Defining search space...")
    parameters = {
        "n_estimators": katib.search.int(min=50, max=100, step=10),  # Smaller range
        "max_depth": katib.search.int(min=5, max=10, step=1),        # Smaller range
    }
    
    print("[DEBUG] Search space:")
    print("  - n_estimators: [50, 100] step 10")
    print("  - max_depth: [5, 10] step 1")
    
    # Initialize Katib client
    client = katib.KatibClient(namespace=katib_namespace)
    
    experiment_name = f"heart-hpo-debug-{int(time.time())}"
    
    print(f"\n[DEBUG] Submitting experiment: {experiment_name}")
    print(f"[DEBUG] Max trials: {max_trials}")
    print(f"[DEBUG] Parallel trials: {parallel_trials}")
    
    try:
        # Submit experiment
        client.tune(
            name=experiment_name,
            objective=train_heart_disease_simple,
            parameters=parameters,
            objective_metric_name="accuracy",
            objective_type="maximize",
            max_trial_count=max_trials,
            parallel_trial_count=parallel_trials,
            resources_per_trial={
                "cpu": "100m",      # Very low resources for debugging
                "memory": "256Mi"
            }
        )
        
        print(f"\n[DEBUG] Experiment submitted!")
        print(f"[DEBUG] Check experiment status:")
        print(f"  kubectl get experiment {experiment_name} -n {katib_namespace}")
        print(f"[DEBUG] Check trials:")
        print(f"  kubectl get trials -n {katib_namespace} -l experiment={experiment_name}")
        
        # Wait with shorter timeout
        print(f"\n[DEBUG] Waiting for completion (timeout: 3 minutes)...")
        
        client.wait_for_experiment_condition(
            name=experiment_name,
            namespace=katib_namespace,
            timeout=180  # 3 minutes only
        )
        
        print(f"\n[DEBUG] Experiment completed!")
        
        # Get best hyperparameters
        best_params = client.get_optimal_hyperparameters(experiment_name)
        
        print(f"\n[DEBUG] Best hyperparameters: {best_params}")
        
        best_n_est = 100  # Default
        best_depth = 10   # Default
        best_acc = 0.85   # Default
        
        if best_params:
            # Extract from parameter_assignments
            if 'parameter_assignments' in best_params and best_params['parameter_assignments']:
                for param in best_params['parameter_assignments']:
                    if param['name'] == "n_estimators":
                        best_n_est = int(param['value'])
                    elif param['name'] == "max_depth":
                        best_depth = int(param['value'])
            
            # Extract accuracy from observation metrics
            if 'observation' in best_params and best_params['observation']:
                if 'metrics' in best_params['observation'] and best_params['observation']['metrics']:
                    for metric in best_params['observation']['metrics']:
                        if metric['name'] == "accuracy":
                            best_acc = float(metric['latest'])
        
        print(f"\n✅ Katib Results:")
        print(f"  Best n_estimators: {best_n_est}")
        print(f"  Best max_depth: {best_depth}")
        print(f"  Best accuracy: {best_acc:.4f}")
        
    except Exception as e:
        print(f"\n❌ Katib experiment failed!")
        print(f"Error: {e}")
        print(f"\nDEBUG COMMANDS:")
        print(f"  kubectl get experiment {experiment_name} -n {katib_namespace} -o yaml")
        print(f"  kubectl get trials -n {katib_namespace} -l experiment={experiment_name}")
        print(f"  kubectl logs -n {katib_namespace} <trial-name> --all-containers")
        print(f"\nUsing fallback hyperparameters...")
        
        best_n_est = 100
        best_depth = 10
        best_acc = 0.80
    
    Outputs = namedtuple('Outputs', ['best_n_estimators', 'best_max_depth', 'best_accuracy'])
    return Outputs(best_n_est, best_depth, best_acc)


# ============================================================================
# COMPONENT 4: Model Training
# ============================================================================
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    input_data: Input[Artifact],
    best_n_estimators: int,
    best_max_depth: int,
    model_output: Output[Artifact]
) -> NamedTuple('Outputs', [('train_accuracy', float), ('test_accuracy', float)]):
    """Train model"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    from collections import namedtuple
    
    print("COMPONENT 4: MODEL TRAINING")
    
    df = pd.read_csv(input_data.path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    joblib.dump(model, model_output.path)
    
    Outputs = namedtuple('Outputs', ['train_accuracy', 'test_accuracy'])
    return Outputs(float(train_acc), float(test_acc))


# ============================================================================
# COMPONENT 5: Model Evaluation
# ============================================================================
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def evaluate_model(
    input_data: Input[Artifact],
    model: Input[Artifact],
    evaluation_report: Output[Artifact]
) -> NamedTuple('Outputs', [('accuracy', float), ('f1_score', float)]):
    """Evaluate model"""
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import json
    from collections import namedtuple
    
    print("COMPONENT 5: MODEL EVALUATION")
    
    clf = joblib.load(model.path)
    df = pd.read_csv(input_data.path)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    report = {
        'accuracy': float(accuracy),
        'f1_score': float(f1)
    }
    
    with open(evaluation_report.path, 'w') as f:
        json.dump(report, f, indent=2)
    
    Outputs = namedtuple('Outputs', ['accuracy', 'f1_score'])
    return Outputs(float(accuracy), float(f1))


# ============================================================================
# COMPONENT 6: Model Registry
# ============================================================================
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["boto3", "joblib"]
)
def register_model(
    model: Input[Artifact],
    accuracy: float,
    f1_score: float,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str
) -> str:
    """Register model to MinIO"""
    import boto3
    from datetime import datetime
    import json
    
    print("COMPONENT 6: MODEL REGISTRY")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        verify=False
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"v_{timestamp}_acc_{accuracy:.4f}".replace(".", "_")
    model_key = f"models/heart-disease/{model_version}/model.joblib"
    
    with open(model.path, 'rb') as f:
        s3_client.upload_fileobj(f, bucket_name, model_key)
    
    metadata = {
        'version': model_version,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'timestamp': timestamp
    }
    
    metadata_key = f"models/heart-disease/{model_version}/metadata.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata).encode()
    )
    
    model_uri = f"s3://{bucket_name}/{model_key}"
    print(f"✅ Model registered: {model_uri}")
    
    return model_uri


# ============================================================================
# PIPELINE
# ============================================================================
@dsl.pipeline(
    name="heart-disease-katib-debug",
    description="DEBUG version with 3 trials and detailed logging"
)
def heart_disease_pipeline(
    minio_endpoint: str = "http://192.168.2.4:19000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
    bucket_name: str = "ml-data",
    data_prefix: str = "raw/heart-disease",
    katib_namespace: str = "kubeflow-user-example-com"
):
    prepare_task = prepare_data(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        data_prefix=data_prefix
    )
    
    validate_task = validate_data(
        input_data=prepare_task.outputs['output_data']
    )
    
    katib_task = katib_hyperparameter_tuning_debug(
        input_data=prepare_task.outputs['output_data'],
        katib_namespace=katib_namespace,
        max_trials=3,
        parallel_trials=1
    )
    katib_task.after(validate_task)
    
    train_task = train_model(
        input_data=prepare_task.outputs['output_data'],
        best_n_estimators=katib_task.outputs['best_n_estimators'],
        best_max_depth=katib_task.outputs['best_max_depth']
    )
    
    eval_task = evaluate_model(
        input_data=prepare_task.outputs['output_data'],
        model=train_task.outputs['model_output']
    )
    
    register_task = register_model(
        model=train_task.outputs['model_output'],
        accuracy=eval_task.outputs['accuracy'],
        f1_score=eval_task.outputs['f1_score'],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name
    )


if __name__ == "__main__":
    output_file = "heart_disease_pipeline_katib_debug.yaml"
    
    print("=" * 70)
    print("COMPILING DEBUG PIPELINE")
    print("=" * 70)
    
    compiler.Compiler().compile(
        pipeline_func=heart_disease_pipeline,
        package_path=output_file
    )
    
    print(f"\n✅ Pipeline compiled: {output_file}")
    print("\n🔧 DEBUG Configuration:")
    print("  - Only 3 trials (faster)")
    print("  - 1 parallel trial (sequential for easier debugging)")
    print("  - 3 minute timeout")
    print("  - Minimal resources (100m CPU, 256Mi RAM)")
    print("  - Simple training function (no actual training)")
    print("  - Detailed logging")
    print("\n📝 After running, check:")
    print("  kubectl get experiment -n kubeflow-user-example-com")
    print("  kubectl get trials -n kubeflow-user-example-com")
    print("=" * 70)