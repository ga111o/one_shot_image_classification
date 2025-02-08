import os
from ultralytics import YOLO
from PIL import Image
import cv2

def train_model():
    model = YOLO('yolo11n.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=800,
        batch=16,
        name='wikiart_detection',
        save_period=1,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        patience=10,
    )

if __name__ == "__main__":
    train_model()
    