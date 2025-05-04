import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir='images', train_ratio=0.8):
    train_dir = Path('train')
    val_dir = Path('val')
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files) 
    
   
    split_point = int(len(image_files) * train_ratio)
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]
    
   
    for f in train_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(train_dir, f)
        shutil.copy2(src, dst)
        
    for f in val_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(val_dir, f)
        shutil.copy2(src, dst)
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

if __name__ == "__main__":
    split_dataset()
