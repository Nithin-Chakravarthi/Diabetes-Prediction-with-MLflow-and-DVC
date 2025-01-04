import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Importing the Random Forest Classifier
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Evaluation metrics
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from mlflow.models import infer_signature  # To infer model signature for MLflow
import os
import pickle
import yaml
from urllib.parse import urlparse  # For parsing URLs
import mlflow
import mlflow.sklearn  # MLflow integration with sklearn models

# Set environment variables for MLflow tracking
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/nithinchakravarthi236/machinelearningpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "nithinchakravarthi236"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "3d5650099751e2a9d291ab28245f916c35afb332"

# Function for hyperparameter tuning using GridSearchCV
def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()  # Initialize the model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)  # Perform grid search
    return grid_search

# Load all the training parameters from params.yaml
params = yaml.safe_load(open('params.yaml'))['train']

# Main training function
def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)  # Load the dataset
    X = data.drop(columns=["Outcome"])  # Features
    y = data["Outcome"]  # Target variable

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    with mlflow.start_run():  # Start an MLflow run
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
        signature = infer_signature(X_train, y_train)  # Infer the input/output signature for the model

        # Define hyperparameter grid for tuning
        param_grid = {
            "n_estimators": [n_estimators, 200],
            "max_depth": [max_depth, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_  # Get the best model from grid search

        # Make predictions and evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy

        print(f"Accuracy: {accuracy}")

        # Log metrics and hyperparameters to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_param("best_max_depth", grid_search.best_params_["max_depth"])
        mlflow.log_param("best_min_samples_split", grid_search.best_params_["min_samples_split"])
        mlflow.log_param("best_min_samples_leaf", grid_search.best_params_["min_samples_leaf"])

        # Log the confusion matrix and classification report as artifacts
        cm = confusion_matrix(y_test, y_pred)
        clr = classification_report(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(clr, "classification_report.txt")

        # Determine tracking URL type to save the model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, model_path, registered_model_name="RandomForestModel")
        else:
            mlflow.sklearn.log_model(best_model, model_path, signature=signature)

        # Save the model locally as a pickle file
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        print(f"Model saved at: {model_path}")

# Entry point to execute the training function
if __name__ == "__main__":
    train(params["data"], params["model"], params["random_state"], params["n_estimators"], params["max_depth"])
