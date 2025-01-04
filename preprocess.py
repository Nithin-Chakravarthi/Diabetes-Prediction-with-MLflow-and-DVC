import pandas as pd  # For data manipulation
import yaml  # For loading YAML configuration files
import os  # For file system operations

# Load parameters from params.yaml under the 'preprocess' section
params = yaml.safe_load(open('params.yaml'))['preprocess']

# Function for preprocessing the data
def preprocess(input_path, output_path):
    # Load the input data from the specified path
    data = pd.read_csv(input_path)

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the preprocessed data to the specified output path
    data.to_csv(output_path, header=None, index=False)
    print(f"Preprocessing done. Saved at {output_path}")

# Entry point to execute the preprocessing function
if __name__ == "__main__":
    preprocess(params["input"], params["output"])
