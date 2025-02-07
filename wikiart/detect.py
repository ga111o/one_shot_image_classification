import cv2
from ultralytics import YOLO
import os

def detect_artwork(image_path, output_path):
    model = YOLO('runs/detect/train17/weights/best.pt')
    
    results = model.predict(image_path)
    
    img = cv2.imread(image_path)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    input_folder = 'image_user'
    output_folder = 'image_user_detect'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            detect_artwork(input_path, output_path)
