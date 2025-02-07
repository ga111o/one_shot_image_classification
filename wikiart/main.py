import os
from ultralytics import YOLO
from PIL import Image
import cv2

def train_model():
    model = YOLO('yolo11n.pt')
    
    model.add_callback('on_train_start', lambda: model.model.set_lora(
        r=4,
        alpha=4,
        dropout=0.1,
        module_filter=['conv']
    ))
    
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='artwork_detection',
        lora=True,
        save_period=1,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5, 
    )

if __name__ == "__main__":
    train_model()
    