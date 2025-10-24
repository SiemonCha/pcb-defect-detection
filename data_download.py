"""
Download from Roboflow 100 - curated public datasets
"""
from roboflow import Roboflow

API_KEY = "NdpJQ5HepZGa7Xvw9wQD"

rf = Roboflow(api_key=API_KEY)

# Roboflow 100 is a collection of 100 public datasets
# Try their PCB dataset
workspace = rf.workspace("roboflow-100")
project = workspace.project("printed-circuit-board")
version = project.version(2)

print("==== Downloading Roboflow 100 PCB dataset...")
dataset = version.download("yolov8", location="data")

print(f"==== Downloaded to: {dataset.location}")