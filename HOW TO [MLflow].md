# **MLflow**
**MLflow** is an open-source platform that provides tools for managing the machine learning lifecycle, including model storage, experiment tracking, and deployment.

**Features**:
- **Model Logging**: You can log model parameters, metrics, and artifacts.
- **Experiment Tracking**: Track experiments and compare different model versions.
- **Model Management**: Store and serve models for inference.

**How to Use**:
- You can run **MLflow** locally or set it up on a **self-hosted server** for collaboration, without any cost.
- For storage, you can use **local storage** or connect to free cloud storage options like **AWS S3 (free tier)** or **Google Cloud Storage (free tier)**.

**Installation**:
```bash
pip install mlflow
```

**Example** (log a simple model):
```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Example model training
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier()
model.fit(X, y)

# Log the model with MLflow
mlflow.sklearn.log_model(model, "model")
```
---

# **Steps to Use MLflow with AWS S3**

## **Step 1: Set Up Your AWS Account**
1. **Sign in to AWS**:
   - Go to [AWS Management Console](https://aws.amazon.com/console/) and log in with your **student credentials**.

2. **Create an S3 Bucket**:
   - Navigate to **S3** in the AWS console, and create a new **S3 bucket** to store your models and artifacts.
   - Make sure the bucket has a unique name (e.g., `my-mlflow-models`).
   - Keep the **default settings** unless you need specific configurations like encryption or public access controls.

   **Steps**:
   - In the AWS Console, go to **S3**.
   - Click **Create bucket**.
   - Enter a unique bucket name (e.g., `my-mlflow-models`).
   - Choose **Region** that is nearest to you.
   - Click **Create**.

3. **Configure IAM User Permissions**:
   - To interact with your **S3 bucket** from MLflow, you’ll need to configure an IAM (Identity and Access Management) user with proper permissions.

   **Steps**:
   - Go to the **IAM Console** and create a new IAM user with **programmatic access**.
   - Attach the policy **`AmazonS3FullAccess`** (or create a custom policy with access only to your bucket).
   - Save the **Access Key ID** and **Secret Access Key** that AWS generates for this IAM user. You’ll need them to configure your environment.

---

## **Step 2: Install MLflow and AWS SDK**

1. **Install MLflow**:
   - Install **MLflow** using `pip` (if you haven’t already).

   ```bash
   pip install mlflow
   ```

2. **Install AWS SDK for Python (Boto3)**:
   - The **Boto3** library is used to interact with AWS services like **S3** from Python.

   ```bash
   pip install boto3
   ```

3. **Configure AWS Credentials**:
   - AWS credentials (Access Key ID and Secret Access Key) are required to authenticate your requests from MLflow to **AWS S3**.
   - You can configure your credentials by setting them in the **AWS CLI** or manually using **Boto3**.

   **Configure AWS CLI (optional)**:
   ```bash
   aws configure
   ```

   Enter your **Access Key ID** and **Secret Access Key** when prompted. Choose the default region (or set it to the region of your S3 bucket).

---

## **Step 3: Set Up MLflow to Use S3 as Remote Artifact Store**

1. **Set the MLflow Tracking URI**:
   - To configure **MLflow** to use **Amazon S3** as the backend store for artifacts (models, logs), set the **MLflow tracking URI** to point to your **S3 bucket**.

   You can do this programmatically in your Python script or set it as an environment variable.

   **Example for programmatic configuration**:
   ```python
   import mlflow

   # Set up MLflow to use S3 for model artifact storage
   mlflow.set_tracking_uri("http://mlflow-server")  # Optional, if using an MLflow server

   # Configure remote artifact storage on S3
   artifact_location = "s3://my-mlflow-models/artifacts/"
   mlflow.set_experiment("my-experiment")

   with mlflow.start_run():
       mlflow.log_param("param1", 5)
       mlflow.log_metric("metric1", 0.9)
       mlflow.log_artifact("path/to/artifact")
   ```

2. **Configure AWS Credentials for Boto3**:
   If you haven’t set the **AWS credentials** via the CLI (`aws configure`), you can specify them directly in your code or environment variables.

   **Using Environment Variables**:
   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key-id"
   export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
   ```

   **Using Boto3 (in code)**:
   ```python
   import boto3

   s3 = boto3.client('s3', 
                     aws_access_key_id='your-access-key-id', 
                     aws_secret_access_key='your-secret-access-key', 
                     region_name='us-west-2')  # Set your region
   ```

3. **Configure S3 Artifact Location in MLflow**:
   - Now that **S3** is set up for artifact storage, you can specify it in your `mlflow.set_tracking_uri()` as the artifact store URL. This will store the models, parameters, and metrics in **S3** instead of the local file system.

---

## **Step 4: Train and Log Models with MLflow**

Now that everything is configured, you can start using **MLflow** to train models and save them to **S3**.

1. **Train a Model and Log with MLflow**:
   For example, if you're training a **scikit-learn** model:

   ```python
   from sklearn.ensemble import RandomForestClassifier
   import mlflow
   import mlflow.sklearn

   # Train a random forest model
   model = RandomForestClassifier(n_estimators=100)
   model.fit(X_train, y_train)

   # Log the model to S3
   with mlflow.start_run():
       mlflow.log_param("n_estimators", 100)
       mlflow.sklearn.log_model(model, "model")
   ```

   - **Log Model**: The model will be logged to the **S3** bucket as specified by the `artifact_location`.
   - **MLflow UI**: You can track your experiments, parameters, and models through the **MLflow UI** by running `mlflow ui` in your terminal.

   ```bash
   mlflow ui
   ```

   By default, the UI will be available at `http://127.0.0.1:5000`.

2. **Log Metrics and Artifacts**:
   You can also log other things like metrics, files, or datasets to the same S3 bucket:

   ```python
   with mlflow.start_run():
       mlflow.log_metric("accuracy", 0.95)
       mlflow.log_artifact("model_metrics.txt")
   ```

---

## **Step 5: Access and Use the Saved Models from S3**

To load a model stored in **S3**, use `mlflow.sklearn.load_model` or equivalent for other model types.

Example of loading a saved model:
```python
# Load model from S3
model_uri = "s3://my-mlflow-models/artifacts/model"
loaded_model = mlflow.sklearn.load_model(model_uri)
```

This allows you to **use the trained models** saved in your S3 bucket for predictions or further evaluation.

---