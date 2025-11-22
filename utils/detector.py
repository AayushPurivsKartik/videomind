# utils/detector.py
from ultralytics import YOLO
import cv2
import os

# Load the best free model (auto-downloads ~80MB)
model = YOLO("yolov8x.pt")  # yolov8x = highest accuracy (you can switch to yolov8m or yolov8n for speed)

def detect_and_draw(image_path, output_path=None, conf_threshold=0.3):
    """
    Runs YOLOv8 detection and returns:
    - detections list (for later search)
    - annotated image path (saved if output_path given)
    """
    results = model(image_path, verbose=False)[0]
    
    detections = []
    img = cv2.imread(image_path)
    
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = results.names[cls_id]
        conf = box.conf[0].item()
        
        if conf < conf_threshold:
            continue
            
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        detections.append({
            "label": label,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2]
        })
        
        # Draw on image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
    
    return detections, output_path