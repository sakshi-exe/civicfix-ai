import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in .env")

rf = Roboflow(api_key=api_key)

project = rf.workspace(
    "indian-institute-of-technology-madras-xamot"
).project(
    "pothole-detection-huf2x"
)

dataset = project.version(2).download("yolov8")