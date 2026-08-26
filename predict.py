from ultralytics import YOLO
from pathlib import Path

# ==============================
# CIVICFIX AI - POTHOLE DETECTOR
# ==============================

MODEL_PATH = "Pothole-detection-2/runs/detect/train/weights/best.pt"
IMAGE_PATH = "Pothole-detection-2/test/images"

model = YOLO(MODEL_PATH)

print("\n🚧 CIVICFIX AI")
print("=" * 50)
print("AI-powered pothole detection\n")


def calculate_severity(box, image_width, image_height):
    """
    Estimate pothole severity from the detected
    bounding-box area relative to the image area.
    """

    x1, y1, x2, y2 = box

    pothole_width = x2 - x1
    pothole_height = y2 - y1

    pothole_area = pothole_width * pothole_height
    image_area = image_width * image_height

    area_ratio = pothole_area / image_area

    if area_ratio >= 0.15:
        severity = "HIGH"
    elif area_ratio >= 0.05:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return severity, area_ratio


results = model.predict(
    source=IMAGE_PATH,
    conf=0.50,
    iou=0.45,
    save=True,
    device="mps"
)

print("=" * 50)
print("PREDICTION RESULTS")
print("=" * 50)

total_potholes = 0

for result in results:

    image_name = Path(result.path).name

    if result.boxes is None or len(result.boxes) == 0:
        print(f"\n📷 {image_name}")
        print("   ❌ No pothole detected")
        continue

    image_height, image_width = result.orig_shape

    potholes_in_image = 0

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        if class_name.lower() != "pothole":
            continue

        coordinates = box.xyxy[0].tolist()

        severity, area_ratio = calculate_severity(
            coordinates,
            image_width,
            image_height
        )

        potholes_in_image += 1
        total_potholes += 1

        print(f"\n📷 {image_name}")
        print(f"   🕳️ Pothole detected")
        print(f"   🎯 Confidence: {confidence:.2%}")
        print(f"   📐 Area ratio: {area_ratio:.2%}")
        print(f"   🚨 Severity: {severity}")

print("\n" + "=" * 50)
print(f"🕳️ Total detections: {total_potholes}")
print("=" * 50)

print("\n✅ CivicFix AI analysis complete!")
print("📁 Annotated images saved in runs/detect/")