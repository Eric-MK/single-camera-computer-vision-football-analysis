import roboflow 
from config import ROBOFLOW_API_KEY  # Import the API_KEY from config.py

# Initialize Roboflow
rf = roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)

# Access the workspace and project
project = rf.workspace("mihailo").project("football-field-detection-f07vi-apxzb")
# Access the specific version of the dataset
version = project.version(1)
# Download the dataset in YOLOv8 format
dataset = version.download("yolov8")

# Print a success message and the dataset's location
print("Dataset downloaded successfully!")
print("Dataset path:", dataset.location)  # Shows where the dataset is saved