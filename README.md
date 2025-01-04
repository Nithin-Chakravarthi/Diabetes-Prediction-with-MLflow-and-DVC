# Diabetes Prediction Machine Learning Pipeline with MLflow and DVC

## Project Overview
This project implements a complete machine learning pipeline for predicting diabetes using the Pima Indians Diabetes Dataset. The pipeline integrates MLflow for experiment tracking and DVC for data version control, creating a reproducible and production-ready workflow for medical prediction.

## Dataset Description
The Pima Indians Diabetes Dataset contains medical predictor variables and one target variable (Outcome). The dataset includes diagnostic measurements for females of Pima Indian heritage. Here are the features:

1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration at 2 hours in an oral glucose tolerance test
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg/(height in m)²)
7. **DiabetesPedigreeFunction**: A function that scores likelihood of diabetes based on family history
8. **Age**: Age in years
9. **Outcome**: Class variable (0: non-diabetic, 1: diabetic)

### Data Format Example
```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
4,148,72,35,0,28.0,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,55,23,94,28.1,0.167,21,0
```

## Features
- Automated data preprocessing pipeline with handling for missing values (zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI)
- Hyperparameter tuning using GridSearchCV optimized for diabetes prediction
- Random Forest Classifier training with feature importance analysis
- Model evaluation focused on medical prediction metrics
- Integration with MLflow for experiment tracking
- Version control for medical data and models using DVC
- Reproducible pipeline stages with clear dependencies

## Technical Architecture
The project is structured into three main pipeline stages:
1. **Preprocessing**: 
   - Handles missing value imputation
   - Normalizes numerical features
   - Performs feature scaling for medical measurements
   
2. **Training**: 
   - Implements Random Forest Classifier with medical prediction optimization
   - Performs stratified cross-validation for balanced class handling
   - Includes feature importance analysis
   
3. **Evaluation**: 
   - Calculates medical-specific metrics (sensitivity, specificity)
   - Performs model evaluation with emphasis on false negative reduction
   - Generates ROC curves and precision-recall curves

## Project Structure
```
├── data/
│   ├── raw/           # Original Pima Indians Diabetes Dataset
│   └── processed/     # Preprocessed and scaled medical data
├── models/            # Trained diabetes prediction models
├── src/
│   ├── __init__.py
│   ├── preprocess.py  # Medical data preprocessing
│   ├── train.py      # Model training with medical focus
│   └── evaluate.py   # Medical-specific evaluation metrics
├── params.yaml        # Configuration parameters
└── dvc.yaml          # DVC pipeline configuration
```

## Prerequisites
- Python 3.x
- MLflow
- DVC (Data Version Control)
- pandas
- scikit-learn
- PyYAML
- numpy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd diabetes-prediction-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure MLflow:
```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/your-username/your-repo.mlflow"
export MLFLOW_TRACKING_USERNAME="your-username"
export MLFLOW_TRACKING_PASSWORD="your-password"
```

## Configuration
The pipeline parameters in `params.yaml` are optimized for diabetes prediction:

```yaml
preprocess:
  input: data/raw/diabetes.csv
  output: data/processed/diabetes_processed.csv
  # Additional preprocessing parameters for medical data
  glucose_threshold: 70
  bmi_threshold: 18.5

train:
  data: data/processed/diabetes_processed.csv
  model: models/diabetes_model.pkl
  random_state: 42
  n_estimators: 100
  max_depth: 10
  # Class weight parameters for imbalanced medical data
  class_weight: balanced

evaluate:
  # Input paths for evaluation
  data: data/processed/diabetes_processed.csv
  model: models/diabetes_model.pkl
  
  # Output paths for evaluation results
  metrics_file: reports/metrics.json
  confusion_matrix_path: reports/confusion_matrix.png
  roc_curve_path: reports/roc_curve.png
  
  # Evaluation parameters
  prediction_threshold: 0.5  # Threshold for converting probabilities to class predictions
  cv_folds: 5  # Number of cross-validation folds
  
  # Metrics to calculate
  metrics:
    - accuracy
    - precision
    - recall
    - f1
    - roc_auc
    - average_precision
  
  # Medical-specific thresholds and settings
  clinical_sensitivity_threshold: 0.9  # Minimum required sensitivity for medical screening
  specificity_target: 0.8  # Target specificity for medical diagnosis
```

## Model Details
The Random Forest Classifier is specifically tuned for diabetes prediction:
- Optimized for handling imbalanced medical data
- Feature importance analysis for medical predictors
- Threshold optimization for clinical decision-making
- Comprehensive metrics including:
  - Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)
  - ROC-AUC Score
  - Precision-Recall curves

## MLflow Integration
The project tracks medical-specific metrics in MLflow:
- Feature importance rankings
- Confusion matrix with medical interpretation
- ROC curves and AUC scores
- Prediction threshold optimization results
- Cross-validation performance on medical metrics

## Pipeline Stages in Detail

### 1. Preprocessing (`preprocess.py`)
- Handles missing values in medical measurements
- Implements domain-specific data validation
- Performs feature scaling appropriate for medical data
- Generates statistical summaries of processed data

### 2. Training (`train.py`)
- Implements stratified sampling for balanced training
- Performs medical-specific feature selection
- Optimizes model for both sensitivity and specificity
- Includes clinical threshold optimization

### 3. Evaluation (`evaluate.py`)
- Calculates medical-specific performance metrics
- Generates clinical performance reports
- Provides feature importance analysis
- Creates visualization of model performance

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


