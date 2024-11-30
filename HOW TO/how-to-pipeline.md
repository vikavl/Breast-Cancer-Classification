# **Pipeline: Classification with CNN and Boosting**

## **1. Data Preparation for Modeling**
- **Task**: Split the dataset into **training**, **validation**, and **test sets**.
- **Details**:
  - Use **stratified sampling** to ensure class balance in all splits.
  - Evaluate whether **cross-validation (CV)** is necessary for the dataset. CV is more relevant for boosting methods to tune hyperparameters.

```python
from sklearn.model_selection import train_test_split

# Example dataset paths and labels
X = metadata['image_path']  # Paths to images
y = metadata['label']  # Labels: Benign or Malignant

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
```

---

## **2. Data Augmentation Techniques**
- **Task**: Apply **data augmentation** to increase dataset diversity and reduce overfitting risks.
- **Key Points**:
  - Overuse of augmentation may cause **overfitting** (use dropout layers in CNN to mitigate this).
  - Aim to expand the dataset while maintaining feature consistency.

```python
from keras.preprocessing.image import ImageDataGenerator

# Define data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation/test data generators (no augmentation, just rescaling)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Flow training data
train_generator = train_datagen.flow_from_dataframe(
    dataframe=metadata.loc[metadata.index.isin(X_train.index)],
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = test_datagen.flow_from_dataframe(
    dataframe=metadata.loc[metadata.index.isin(X_val.index)],
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```

---

## **3. CNN (Custom and Premade Models)**

### **3.1 Custom CNN**
- Build a custom CNN architecture for the task.
- Include **Dropout layers** to reduce overfitting (as discussed in your plan).

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define a custom CNN
custom_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
custom_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### **3.2 Pretrained CNN (Transfer Learning)**
- Use **transfer learning** with pretrained CNNs (e.g., **VGG16**, **ResNet50**) for feature extraction or fine-tuning.

```python
from keras.applications import ResNet50
from keras.models import Model

# Load ResNet50 pretrained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for binary classification
premade_cnn = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

premade_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## **4. Boosting Methods**
- Train **boosting models** (e.g., **XGBoost**, **LightGBM**) using features extracted from the **CNN** (custom or pretrained).
- **Steps**:
  - Extract features using the **intermediate CNN layer**.
  - Train the boosting model with the extracted features.

```python
from xgboost import XGBClassifier

# Extract features from CNN for boosting
feature_extractor = Model(inputs=custom_cnn.input, outputs=custom_cnn.layers[-2].output)
X_train_features = feature_extractor.predict(train_generator)
X_val_features = feature_extractor.predict(val_generator)

# Train XGBoost
xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
xgb_model.fit(X_train_features, y_train)

# Evaluate boosting model
xgb_val_accuracy = xgb_model.score(X_val_features, y_val)
print(f"XGBoost Validation Accuracy: {xgb_val_accuracy}")
```

---

## **5. Embedding Methods**
- Combine **structured data** (e.g., metadata like patient age, lesion size) with CNN outputs to improve classification.

```python
# Example: Combine metadata with CNN features
metadata_features = metadata[['age', 'lesion_size']].values
combined_train_features = np.concatenate((X_train_features, metadata_features), axis=1)

# Train boosting model on combined features
xgb_combined = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)
xgb_combined.fit(combined_train_features, y_train)
```

---

## **6. MLflow Integration**

Log all experiments (CNN, boosting, embedding) into **MLflow** for tracking and reproducibility.

```python
import mlflow
import mlflow.keras
import mlflow.xgboost

# Start MLflow run for custom CNN
with mlflow.start_run():
    # Log CNN parameters and metrics
    mlflow.log_param("cnn_optimizer", "adam")
    mlflow.log_param("cnn_dropout", 0.5)
    mlflow.log_metric("cnn_val_accuracy", custom_cnn.evaluate(val_generator)[1])
    mlflow.keras.log_model(custom_cnn, "custom_cnn")

# Start MLflow run for boosting
with mlflow.start_run():
    mlflow.log_param("xgb_n_estimators", 100)
    mlflow.log_metric("xgb_val_accuracy", xgb_val_accuracy)
    mlflow.xgboost.log_model(xgb_model, "xgb_model")
```

---

### **Final Pipeline Overview**

1. **Data Preparation**:
   - Split data into training, validation, and test sets.
   - Apply data augmentation for the training set.

2. **CNN Models**:
   - Train both **custom CNN** and **pretrained CNNs** (e.g., ResNet50, VGG16).
   - Fine-tune the CNN models for breast cancer classification.

3. **Boosting Models**:
   - Use **CNN-extracted features** to train boosting models (e.g., XGBoost, LightGBM).
   - Experiment with combining structured metadata (e.g., age, lesion size) with CNN features.

4. **Embedding Methods**:
   - Merge structured and unstructured data into a unified feature representation.
   - Train boosting models on these combined features.

5. **MLflow Integration**:
   - Log all experiments, hyperparameters, and metrics.
   - Save models (CNN and boosting) to a remote storage backend (e.g., AWS S3).