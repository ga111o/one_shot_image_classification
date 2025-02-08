import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ImageProcessingLogger')

def process_image(image_path, output_path, label_path, new_label_path):
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read image: {image_path}")
        return
    h, w = img.shape[:2]
    
    r = random.randint(200, 255)
    g = random.randint(200, 255)
    b = random.randint(200, 255)
    background = np.ones((800, 800, 3), dtype=np.uint8)
    background[:, :, 0] = b
    background[:, :, 1] = g
    background[:, :, 2] = r
    
    max_size = 500
    scale = min(max_size/w, max_size/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized_img = cv2.resize(img, (new_w, new_h))
    
    shear_factor_x = np.random.uniform(-0.15, 0.15)
    shear_factor_y = np.random.uniform(-0.15, 0.15)
    
    extra_width = int(abs(shear_factor_x) * new_h * 2.5)
    extra_height = int(abs(shear_factor_y) * new_w * 2.5)
    output_w = min(new_w + extra_width * 2, 750)
    output_h = min(new_h + extra_height * 2, 750)
    
    if new_h > output_h or new_w > output_w:
        scale_adjust = min(output_h/new_h, output_w/new_w) * 0.95
        new_w = int(new_w * scale_adjust)
        new_h = int(new_h * scale_adjust)
        resized_img = cv2.resize(img, (new_w, new_h))
    
    padded_img = np.ones((output_h, output_w, 3), dtype=np.uint8)
    padded_img[:, :, 0] = b
    padded_img[:, :, 1] = g
    padded_img[:, :, 2] = r
    
    y_start = (output_h - new_h) // 2
    x_start = (output_w - new_w) // 2
    padded_img[y_start:y_start+new_h, x_start:x_start+new_w] = resized_img
    
    M = np.float32([
        [1, shear_factor_x, 0],
        [shear_factor_y, 1, 0]
    ])
    
    sheared_img = cv2.warpAffine(padded_img, M, (output_w, output_h), borderMode=cv2.BORDER_REPLICATE)
    
    max_x_offset = 800 - output_w
    max_y_offset = 800 - output_h
    x_offset = random.randint(0, max_x_offset)
    y_offset = random.randint(0, max_y_offset)
    background[y_offset:y_offset+output_h, x_offset:x_offset+output_w] = sheared_img
    
    cv2.imwrite(output_path, background)
    
    if os.path.exists(label_path):
        logger.info(f"Updating label file: {label_path}")
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        new_labels = []
        for label in labels:
            class_id, x, y, w, h = map(float, label.strip().split())
            
            px = x * w
            py = y * h
            
            new_px = px + shear_factor_x * py
            new_py = shear_factor_y * px + py
            
            new_x = (new_px * scale + x_offset) / 800
            new_y = (new_py * scale + y_offset) / 800
            new_w = (w * scale) / 800
            new_h = (h * scale) / 800
            
            new_x = max(0, min(1, new_x))
            new_y = max(0, min(1, new_y))
            new_w = max(0, min(1, new_w))
            new_h = max(0, min(1, new_h))
            
            new_labels.append(f"{int(class_id)} {new_x} {new_y} {new_w} {new_h}\n")
        
        with open(new_label_path, 'w') as f:
            f.writelines(new_labels)
        logger.info(f"Processed image and new label saved to: {output_path}, {new_label_path}")

def preprocess_dataset(source_dir, train_ratio=0.8):
    """
    1. wikiart 폴더에서 이미지 추출
    2. train/val 분할
    3. YOLO 라벨 생성
    4. 이미지 처리 및 증강
    """
    logger.info("Starting dataset preprocessing...")
    temp_dir = Path("temp_images")
    train_dir = Path("dataset/train")
    val_dir = Path("dataset/val")
    processed_train_dir = Path("processed_dataset/train")
    processed_val_dir = Path("processed_dataset/val")
    
    for dir_path in [temp_dir, train_dir, val_dir, processed_train_dir, processed_val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Collecting images from wikiart...")
    logger.info("Collecting images from wikiart...")
    image_files = []
    for root, _, files in os.walk(source_dir):
        logger.info(f"Walking through directory: {root}")
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
    logger.info("Splitting dataset...")
    random.shuffle(image_files)
    split_point = int(len(image_files) * train_ratio)
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]
    
    logger.info(f"Train files: {len(train_files)}, Validation files: {len(val_files)}")
    for f in train_files:
        src = os.path.join(temp_dir, f)
        dst = os.path.join(train_dir, f)
        shutil.copy2(src, dst)
        
    for f in val_files:
        src = os.path.join(temp_dir, f)
        dst = os.path.join(val_dir, f)
        shutil.copy2(src, dst)
    
    print("\nStep 3: Creating YOLO labels...")
    logger.info("Creating YOLO labels...")
    for split_dir in [train_dir, val_dir]:
        logger.info(f"Processing split directory: {split_dir}")
        for img_file in os.listdir(split_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                label_path = os.path.join(split_dir, Path(img_file).stem + '.txt')
                with open(label_path, 'w') as f:
                    f.write('0 0.5 0.5 1.0 1.0\n')

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    
    print("\nStep 4: Processing and augmenting images...")
    logger.info("Processing and augmenting images...")
    for split_dir, processed_dir in [(train_dir, processed_train_dir), (val_dir, processed_val_dir)]:
        logger.info(f"Processing split directory: {split_dir}")
        for img_file in os.listdir(split_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(split_dir, img_file)
                output_path = os.path.join(processed_dir, img_file)
                label_name = Path(img_file).stem + '.txt'
                label_path = os.path.join(split_dir, label_name)
                new_label_path = os.path.join(processed_dir, label_name)
                
                process_image(image_path, output_path, label_path, new_label_path)

    shutil.rmtree(temp_dir)
    
    logger.info("Preprocessing completed!")
    logger.info(f"Total images: {len(image_files)}")
    logger.info(f"Training images: {len(train_files)}")
    logger.info(f"Validation images: {len(val_files)}")

if __name__ == "__main__":
    preprocess_dataset(source_dir="wikiart") 