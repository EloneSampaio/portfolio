# Classification Model Evaluation

## Overview
This project provides a complete pipeline for loading datasets, performing cross-validation, selecting the best hyperparameters, training final classification models, and evaluating their performance. It supports multiple file formats (`.npz`, `.csv`, `.json`, `.xlsx`, `.parquet`) and includes a comparison between **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)** classifiers.

## Features
- Load datasets from various formats
- Perform cross-validation
- Evaluate multiple models and hyperparameters
- Select the best-performing model
- Train and evaluate the final model

## Installation
Ensure you have the necessary Python libraries installed:

```bash
pip install numpy pandas scikit-learn openpyxl pyarrow
```

## Usage

### 1. Load Dataset
The script provides two functions for loading datasets:

- `load_data(train_filepath, test_filepath)`: Loads training and test datasets from multiple file formats.
- `load_data2(filepath)`: Loads a single dataset or defaults to the Iris dataset if no file path is provided.

### 2. Perform Cross-Validation
The `CrossValidation` class performs k-fold validation and computes key evaluation metrics:

```python
cv = CrossValidation(n_splits=5)
metrics = cv.evaluate_model(model, X, y)
print(metrics)
```

### 3. Train and Evaluate Models
The `ModelTrainer` class trains the best model with optimized hyperparameters:

```python
trainer = ModelTrainer(model, best_params)
trainer.train_final_model(X_train, y_train)
final_metrics = trainer.evaluate_final_model(X_test, y_test)
print(final_metrics)
```

### 4. Run Experiment End-to-End
The `Experiment` class runs the full pipeline:

```python
experiment = Experiment(train_file, test_file)
experiment.run_experiment()
```

## Example Usage

```python
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# Define file paths for training and testing datasets
train_file = "path/to/train_data.npz"
test_file = "path/to/test_data.npz"

# Run experiment
experiment = Experiment(train_file, test_file)
experiment.run_experiment()
```

## Evaluation Metrics
The models are evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC (if applicable)

## License
This project is open-source and available for modification and distribution.

## Author
[Elone Izata Gonçalves Sampaio](https://www.linkedin.com/in/elonesampaio/)

