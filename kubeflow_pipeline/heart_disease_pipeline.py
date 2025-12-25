from kfp import dsl, compiler
from kfp.dsl import Artifact, Input, Output
from typing import NamedTuple


# ============================================================================
# COMPONENT 1: Data Preparation
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
    """Collect + Preprocess + Feature Engineering"""
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
    
    print(f"Found {len(parquet_files)} parquet files")
    
    dfs = []
    for file_key in parquet_files[:10]:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
        tmp_file_path = tmp_file.name
        print(f"Downloading file: {file_key} to {tmp_file_path}")
        tmp_file.close()
        s3_client.download_file(bucket_name, file_key, tmp_file_path)
        dfs.append(pd.read_parquet(tmp_file_path))
        os.unlink(tmp_file_path)
    
    if not dfs:
        raise Exception("No data found in MinIO!")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined data: {combined_df.shape}")
    
    # Preprocess
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                   'ca', 'thal', 'target']
    df_clean = combined_df[feature_cols].copy()
    df_clean = df_clean.fillna(df_clean.median())
    df_clean = df_clean.drop_duplicates()
    
    # Remove outliers
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_clean[col] = df_clean[col].clip(Q1 - 3*IQR, Q3 + 3*IQR)
    
    # Feature engineering
    df_clean['age_x_thalach'] = df_clean['age'] * df_clean['thalach']
    df_clean['bmi_proxy'] = df_clean['chol'] / (df_clean['thalach'] + 1)
    
    target = df_clean['target']
    features = df_clean.drop('target', axis=1)
    
    # Standardization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_df = pd.DataFrame(features_scaled, columns=features.columns)
    
    df_final = features_df.copy()
    df_final['target'] = target.values
    
    df_final.to_csv(output_data.path, index=False)
    
    print(f"[OK] Complete! {len(df_final)} records, {len(features.columns)} features")
    
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
    
    print("=" * 70)
    print("COMPONENT 2: DATA VALIDATION")
    print("=" * 70)
    
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
    
    print(f"Total records: {len(df)}")
    print(f"Total features: {len(df.columns) - 1}")
    print(f"Result: {'[OK] VALID' if is_valid else '[ERROR] INVALID'}")
    
    if not is_valid:
        raise Exception(f"Validation failed: {validation_report}")
    
    return validation_report


# ============================================================================
# COMPONENT 3: Train and Evaluate Model
# ============================================================================
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "joblib", "numpy"]
)
def train_model(
    input_data: Input[Artifact],
    model_output: Output[Artifact],
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2
) -> NamedTuple('Outputs', [
    ('train_accuracy', float), 
    ('test_accuracy', float),
    ('eval_accuracy', float),
    ('eval_f1_score', float)
]):
    """
    Train and Evaluate Random Forest model with FIXED hyperparameters
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    import joblib
    from collections import namedtuple
    import time
    
    print("=" * 70)
    print("COMPONENT 3: TRAINING AND EVALUATION")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(input_data.path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Hyperparameters
    print(f"\nHyperparameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth: {max_depth}")
    print(f"  min_samples_split: {min_samples_split}")
    print(f"  min_samples_leaf: {min_samples_leaf}")
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Train
    print("\nTraining model...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"[OK] Training completed in {elapsed_time:.2f} seconds")
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Training metrics
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Evaluation metrics (on test set)
    eval_acc = accuracy_score(y_test, test_pred)
    eval_f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)
    
    print(f"\nTraining Results:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    print(f"\nEvaluation Results:")
    print(f"  Accuracy:  {eval_acc:.4f}")
    print(f"  F1-Score:  {eval_f1:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    joblib.dump(model, model_output.path)
    print(f"\n[OK] Model saved to: {model_output.path}")
    
    Outputs = namedtuple('Outputs', ['train_accuracy', 'test_accuracy', 'eval_accuracy', 'eval_f1_score'])
    return Outputs(float(train_acc), float(test_acc), float(eval_acc), float(eval_f1))


# ============================================================================
# COMPONENT 4: Register Model to MinIO
# ============================================================================
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["boto3", "joblib"]
)
def register_model(
    model: Input[Artifact],
    train_accuracy: float,
    test_accuracy: float,
    eval_accuracy: float,
    eval_f1_score: float,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    n_estimators: int,
    max_depth: int
) -> str:
    """Register trained model to MinIO with metadata"""
    import boto3
    from datetime import datetime
    import json
    
    print("=" * 70)
    print("COMPONENT 5: MODEL REGISTRY")
    print("=" * 70)
    
    s3_client = boto3.client(
        's3',
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        verify=False
    )
    
    # Generate version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"v_{timestamp}_acc_{eval_accuracy:.4f}".replace(".", "_")
    
    print(f"Model version: {model_version}")
    
    # Upload model
    model_key = f"models/heart-disease/{model_version}/model.joblib"
    with open(model.path, 'rb') as f:
        s3_client.upload_fileobj(f, bucket_name, model_key)
    print(f"[OK] Model uploaded: s3://{bucket_name}/{model_key}")
    
    # Create metadata
    metadata = {
        'model_version': model_version,
        'timestamp': timestamp,
        'hyperparameters': {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        },
        'metrics': {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'eval_accuracy': float(eval_accuracy),
            'eval_f1_score': float(eval_f1_score)
        },
        'model_type': 'RandomForestClassifier',
        'framework': 'scikit-learn',
        'dataset': 'heart-disease'
    }
    
    # Upload metadata
    metadata_key = f"models/heart-disease/{model_version}/metadata.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode()
    )
    print(f"[OK] Metadata uploaded: s3://{bucket_name}/{metadata_key}")
    
    model_uri = f"s3://{bucket_name}/{model_key}"
    
    print(f"\n[OK] Model registration complete!")
    print(f"Model URI: {model_uri}")
    print(f"Test Accuracy: {eval_accuracy:.4f}")
    print(f"F1-Score: {eval_f1_score:.4f}")
    
    return model_uri


# ============================================================================
# PIPELINE
# ============================================================================
@dsl.pipeline(
    name="heart-disease-simple-no-tuning",
    description="Simple pipeline: Prepare -> Validate -> Train and Evaluate -> Register"
)
def heart_disease_pipeline(
    # MinIO configs
    minio_endpoint: str = "http://192.168.2.4:19000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
    bucket_name: str = "ml-data",
    data_prefix: str = "raw/heart-disease",
    
    # Model hyperparameters (FIXED - no tuning)
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2
):

    
    # ========== STEP 1: DATA PREPARATION ==========
    prepare_task = prepare_data(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        data_prefix=data_prefix
    )
    prepare_task.set_display_name('Data Preparation')
    
    # ========== STEP 2: DATA VALIDATION ==========
    validate_task = validate_data(
        input_data=prepare_task.outputs['output_data']
    )
    validate_task.set_display_name('Data Validation')
    validate_task.after(prepare_task)
    
    # ========== STEP 3: TRAINING AND EVALUATION ==========
    train_task = train_model(
        input_data=prepare_task.outputs['output_data'],
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    train_task.set_display_name('Training and Evaluation')
    train_task.after(validate_task)
    
    # ========== STEP 4: MODEL REGISTRY ==========
    register_task = register_model(
        model=train_task.outputs['model_output'],
        train_accuracy=train_task.outputs['train_accuracy'],
        test_accuracy=train_task.outputs['test_accuracy'],
        eval_accuracy=train_task.outputs['eval_accuracy'],
        eval_f1_score=train_task.outputs['eval_f1_score'],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    register_task.set_display_name('Model Registry')
    register_task.after(train_task)


# ============================================================================
# COMPILE PIPELINE
# ============================================================================
if __name__ == "__main__":
    output_file = "heart_disease_pipeline_simple.yaml"
    
    print("=" * 70)
    print("COMPILING SIMPLE PIPELINE (NO HYPERPARAMETER TUNING)")
    print("=" * 70)
    
    compiler.Compiler().compile(
        pipeline_func=heart_disease_pipeline,
        package_path=output_file
    )
    
    print(f"\n[OK] Pipeline compiled successfully!")
    print(f"Output file: {output_file}")
    
    print("\nPipeline Flow:")
    print("  1. Data Preparation (MinIO -> CSV)")
    print("  2. Data Validation")
    print("  3. Training and Evaluation (RandomForest with FIXED hyperparameters)")
    print("  4. Model Registry (save to MinIO)")
    
    print("\nDefault Hyperparameters:")
    print("  - n_estimators: 100")
    print("  - max_depth: 10")
    print("  - min_samples_split: 5")
    print("  - min_samples_leaf: 2")
    
    print("\nTo run with custom hyperparameters:")
    print("  Upload to Kubeflow UI and modify parameters in the run")
    
    print("=" * 70)