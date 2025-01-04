import pandas as pd  # For data manipulation
import pickle  # For loading the trained model
from sklearn.metrics import accuracy_score  # For evaluating model accuracy
import yaml  # For loading YAML configuration files
import os  # For file system operations
import mlflow  # For logging metrics and model tracking
from urllib.parse import urlparse  # For parsing MLflow tracking URI

# Set environment variables for MLflow tracking
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/nithinchakravarthi236/machinelearningpipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "nithinchakravarthi236"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "3d5650099751e2a9d291ab28245f916c35afb332"

# Load parameters from params.yaml under the 'train' section
params = yaml.safe_load(open('params.yaml'))['train']

# Function for evaluating the trained model
def evaluate(data_path, model_path):
    # Load the dataset for evaluation
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])  # Features
    y = data["Outcome"]  # Target variable

    # Load the trained model from the specified path
    model = pickle.load(open(model_path, "rb"))

    # Make predictions using the loaded model
    predictions = model.predict(X)

    # Calculate accuracy of the model
    accuracy = accuracy_score(y, predictions)

    # Log the accuracy metric to MLflow
    mlflow.log_metric("accuracy", accuracy)
    
    # Print the accuracy
    print(f"Accuracy: {accuracy}")

# Main execution block
if __name__ == "__main__":
    evaluate(params["data"], params["model"])
