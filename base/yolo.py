from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import torch
from datetime import datetime

def prepare_dataset(img_dir):
    # 클래스명 == 파일이름
    class_names = set()
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            class_name = os.path.splitext(img_file)[0]
            class_names.add(class_name)
    
    classes = sorted(list(class_names))
    
    labels_dir = 'labels'
    os.makedirs(labels_dir, exist_ok=True)
    
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            class_name = os.path.splitext(img_file)[0]
            class_idx = classes.index(class_name)
            
            label_content = f"{class_idx} 0.5 0.5 1.0 1.0" # 중심 기준
            
            label_file = os.path.splitext(img_file)[0] + '.txt'
            with open(os.path.join(labels_dir, label_file), 'w') as f:
                f.write(label_content)
    
    data_yaml = {
        'train': str(Path(img_dir).absolute()),
        'val': str(Path(img_dir).absolute()),
        'nc': len(classes),
        'names': classes
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

def train_model():
    img_dir = './image_base'
    img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    model = YOLO("yolo11n.pt")
    
    results = model.train(
        data='data.yaml',
        epochs=1000,
        imgsz=640,
        batch=img_count,
        name='test',
        device='0' if torch.cuda.is_available() else 'cpu',
        augment=True,
        degrees=40.0,
        scale=0.5,
        translate=0.1,
        shear=0.3,
        perspective=0.3,
        flipud=0.0,
        fliplr=0.5,
        copy_paste=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
    )
    
    current_time = datetime.now()
    save_name = current_time.strftime('./pt/%y%m%d_%H%M_yolo.pt')
    model.save(save_name)
    
    return results

def main():
    try:
        prepare_dataset('./image_base')
        
        results = train_model()
        print("done!")
        
    except Exception as e:
        print(f"---error---\n{str(e)}")

if __name__ == "__main__":
    main()
