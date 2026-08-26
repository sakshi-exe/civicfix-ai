from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from pathlib import Path
import tempfile
import os

app = FastAPI(
    title="CivicFix AI",
    description="AI-powered pothole detection API",
    version="1.0.0"
)

MODEL_PATH = (
    Path(__file__).parent
    / "Pothole-detection-2"
    / "runs"
    / "detect"
    / "train"
    / "weights"
    / "best.pt"
)

model = YOLO(str(MODEL_PATH))


def calculate_risk(confidence, area_ratio, pothole_count):
    score = 0

    # Detection confidence
    if confidence >= 0.80:
        score += 40
    elif confidence >= 0.65:
        score += 30
    else:
        score += 20

    # Visual size
    if area_ratio >= 0.15:
        score += 40
    elif area_ratio >= 0.05:
        score += 25
    else:
        score += 10

    # Multiple potholes
    if pothole_count >= 3:
        score += 20
    elif pothole_count >= 2:
        score += 10

    if score >= 70:
        return "HIGH", score
    elif score >= 45:
        return "MEDIUM", score

    return "LOW", score


@app.get("/")
def root():
    return {
        "service": "CivicFix AI",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "YOLO11",
        "model_loaded": True
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Create temporary image file
    suffix = Path(file.filename or "image.jpg").suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        contents = await file.read()
        temp.write(contents)
        temp_path = temp.name

    try:

        results = model.predict(
            source=temp_path,
            conf=0.50,
            iou=0.45,
            device="mps",
            verbose=False
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:

            return {
                "success": True,
                "pothole_detected": False,
                "pothole_count": 0,
                "message": "No pothole detected"
            }

        image_height, image_width = result.orig_shape

        potholes = []

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name.lower() != "pothole":
                continue

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_width = x2 - x1
            box_height = y2 - y1

            box_area = box_width * box_height
            image_area = image_width * image_height

            area_ratio = box_area / image_area

            potholes.append({
                "confidence": confidence,
                "area_ratio": area_ratio,
                "bounding_box": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            })

        if not potholes:

            return {
                "success": True,
                "pothole_detected": False,
                "pothole_count": 0,
                "message": "No pothole detected"
            }

        primary = max(
            potholes,
            key=lambda x: x["confidence"]
        )

        risk_level, risk_score = calculate_risk(
            primary["confidence"],
            primary["area_ratio"],
            len(potholes)
        )

        return {
            "success": True,
            "pothole_detected": True,
            "pothole_count": len(potholes),
            "confidence": round(
                primary["confidence"],
                3
            ),
            "visual_area_ratio": round(
                primary["area_ratio"],
                4
            ),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "detections": potholes
        }

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)