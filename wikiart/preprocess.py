import os
import shutil
import random
from pathlib import Path

def preprocess_dataset(source_dir="wikiart", train_ratio=0.8):
    """
    1. wikiart 폴더에서 이미지 추출
    2. train/val 분할
    3. YOLO 라벨 생성
    """
    temp_dir = Path("temp_images")
    train_dir = Path("dataset/train")
    val_dir = Path("dataset/val")
    
    for dir_path in [temp_dir, train_dir, val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Collecting images from wikiart...")
    image_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_file = os.path.join(root, file)
                dest_file = os.path.join(temp_dir, file)
                try:
                    shutil.copy2(source_file, dest_file)
                    image_files.append(file)
                except Exception as e:
                    print(f"Error copying {file}: {str(e)}")
    
    print("\nStep 2: Splitting dataset...")
    random.shuffle(image_files)
    split_point = int(len(image_files) * train_ratio)
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]
    
    for f in train_files:
        src = os.path.join(temp_dir, f)
        dst = os.path.join(train_dir, f)
        shutil.copy2(src, dst)
        
    for f in val_files:
        src = os.path.join(temp_dir, f)
        dst = os.path.join(val_dir, f)
        shutil.copy2(src, dst)
    
    print("\nStep 3: Creating YOLO labels...")
    for split_dir in [train_dir, val_dir]:
        for img_file in os.listdir(split_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                label_path = os.path.join(split_dir, Path(img_file).stem + '.txt')
                with open(label_path, 'w') as f:
                    f.write('0 0.5 0.5 1.0 1.0\n')
    
    shutil.rmtree(temp_dir)
    
    print(f"\nPreprocessing completed!")
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

if __name__ == "__main__":
    preprocess_dataset() 