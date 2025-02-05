import os
from pathlib import Path

def create_yolo_labels(dataset_dir):
    for split in ['train', 'val']:
        img_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(img_dir):
            continue
        
        img_extensions = ['.jpg', '.jpeg', '.png']
        
        for img_file in os.listdir(img_dir):
            if any(img_file.lower().endswith(ext) for ext in img_extensions):
                label_path = os.path.join(img_dir, Path(img_file).stem + '.txt')
                
                with open(label_path, 'w') as f:
                    f.write('0 0.5 0.5 1.0 1.0\n')

if __name__ == '__main__':
    dataset_dir = 'dataset'
    create_yolo_labels(dataset_dir)
