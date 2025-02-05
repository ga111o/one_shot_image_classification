import os
from ultralytics import YOLO
from PIL import Image
import cv2

def train_model():
    model = YOLO('yolo11n.pt') 
    
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='artwork_detection'
    )

def detect_artwork(image_path):
    model = YOLO('runs/detect/artwork_detection/weights/best.pt')
    
    results = model.predict(image_path)
    
    img = cv2.imread(image_path)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    cv2.imwrite('detected_artwork.jpg', img)

if __name__ == "__main__":
    train_model()
    detect_artwork('test.jpg')
