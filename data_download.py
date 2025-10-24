"""
Download from Roboflow 100 - curated public datasets
"""
from roboflow import Roboflow
import os

API_KEY = "NdpJQ5HepZGa7Xvw9wQD"

rf = Roboflow(api_key=API_KEY)

# Roboflow 100 is a collection of 100 public datasets
workspace = rf.workspace("roboflow-100")
project = workspace.project("printed-circuit-board")
version = project.version(2)

print("==== Downloading Roboflow 100 PCB dataset...")
dataset = version.download("yolov8", location="data")

print(f"==== Downloaded to: {dataset.location}")

# Verify data.yaml exists
data_yaml = os.path.join(dataset.location, "data.yaml")
if not os.path.exists(data_yaml):
    raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

print(f"==== Verified: {data_yaml}")

# Save path for other scripts
with open('dataset_path.txt', 'w') as f:
    f.write(dataset.location)

print(f"==== Path saved to dataset_path.txt")