import os
import cv2
import datetime as dt
from ultralytics import YOLO  # ✅ YOLOv11

# ========================
# Configuration
# ========================
MODEL_PATH = r"C:\Users\91797\Desktop\Seatbelt Detection\models\best.pt"
INPUT_VIDEO = r"C:\Users\91797\Desktop\Seatbelt Detection\sample\dms1.MP4"
OUTPUT_FILE = (
    "output/test_result_" + dt.datetime.strftime(dt.datetime.now(), "%Y%m%d%H%M%S") + ".mp4"
)

THRESHOLD_SCORE = 0.5
MAX_FRAME_RECORD = 500
SKIP_FRAMES = 1

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

# ========================
# Load YOLOv11
# ========================
print("Loading YOLOv11 model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully!")

# ========================
# Frame Processing
# ========================
def process_frame(frame):
    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if conf > THRESHOLD_SCORE:
                color = COLOR_GREEN if "Seatbelt" in label else COLOR_RED
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

# ========================
# Video Processing Loop
# ========================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise Exception(f"Error: Cannot open video {INPUT_VIDEO}")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
writer = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

print("Analyzing video... Press 'q' to stop early.")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % SKIP_FRAMES == 0:
        processed_frame = process_frame(frame)
        writer.write(processed_frame)
        cv2.imshow("Seatbelt Detection", processed_frame)

    if frame_count > MAX_FRAME_RECORD:
        print("Reached max frame count limit.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Interrupted by user.")
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"✅ Processing complete. Results saved to: {OUTPUT_FILE}")
