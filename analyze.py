from ultralytics import YOLO
from pathlib import Path
import json
import sys


# ==========================================
# CIVICFIX AI — SINGLE IMAGE ANALYZER
# ==========================================

MODEL_PATH = "Pothole-detection-2/runs/detect/train/weights/best.pt"

# Load trained model
model = YOLO(MODEL_PATH)


def calculate_risk(confidence, area_ratio, pothole_count):
    """
    Calculate an AI-based visual risk level.

    This is NOT a measurement of actual pothole depth.
    """

    score = 0

    # Confidence contribution
    if confidence >= 0.80:
        score += 40
    elif confidence >= 0.65:
        score += 30
    else:
        score += 20

    # Visual size contribution
    if area_ratio >= 0.15:
        score += 40
    elif area_ratio >= 0.05:
        score += 25
    else:
        score += 10

    # Multiple potholes increase risk
    if pothole_count >= 3:
        score += 20
    elif pothole_count >= 2:
        score += 10

    if score >= 70:
        return "HIGH", score
    elif score >= 45:
        return "MEDIUM", score
    else:
        return "LOW", score


def analyze_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return

    print("\n🚧 CIVICFIX AI")
    print("=" * 50)
    print(f"📷 Image: {image_path.name}")
    print("🤖 Running AI analysis...\n")

    results = model.predict(
        source=str(image_path),
        conf=0.50,
        iou=0.45,
        save=True,
        device="mps"
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:

        print("❌ NO POTHOLE DETECTED")

        report = {
            "image": image_path.name,
            "pothole_detected": False,
            "pothole_count": 0
        }

        print("\n📋 CivicFix Report:")
        print(json.dumps(report, indent=2))

        return

    image_height, image_width = result.orig_shape

    potholes = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        if class_name.lower() != "pothole":
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        box_width = x2 - x1
        box_height = y2 - y1

        box_area = box_width * box_height
        image_area = image_width * image_height

        area_ratio = box_area / image_area

        potholes.append({
            "confidence": confidence,
            "area_ratio": area_ratio,
            "bounding_box": [x1, y1, x2, y2]
        })

    if not potholes:

        print("❌ NO POTHOLE DETECTED")

        return

    # Highest-confidence detection
    primary = max(
        potholes,
        key=lambda x: x["confidence"]
    )

    risk_level, risk_score = calculate_risk(
        primary["confidence"],
        primary["area_ratio"],
        len(potholes)
    )

    # ==========================================
    # CIVICFIX RESULT
    # ==========================================

    print("🕳️ POTHOLE DETECTED")
    print("-" * 50)

    print(
        f"🎯 Confidence: "
        f"{primary['confidence']:.2%}"
    )

    print(
        f"📐 Visual Area: "
        f"{primary['area_ratio']:.2%}"
    )

    print(
        f"🕳️ Detected Potholes: "
        f"{len(potholes)}"
    )

    print(
        f"🚨 AI Risk Level: "
        f"{risk_level}"
    )

    print(
        f"📊 Risk Score: "
        f"{risk_score}/100"
    )

    report = {
        "image": image_path.name,
        "pothole_detected": True,
        "pothole_count": len(potholes),
        "confidence": round(primary["confidence"], 3),
        "visual_area_ratio": round(
            primary["area_ratio"], 4
        ),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "status": "Ready for CivicFix report"
    }

    print("\n📋 CIVICFIX REPORT")
    print("=" * 50)

    print(json.dumps(
        report,
        indent=2
    ))

    print("\n✅ Analysis complete!")
    print("📁 Annotated image saved by YOLO.")


# ==========================================
# COMMAND LINE
# ==========================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:\n"
            "python analyze.py <image_path>\n"
        )

        sys.exit(1)

    analyze_image(sys.argv[1])