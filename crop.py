import os
from PIL import Image
import numpy as np
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def segment_image(image_path, folder_path):
    model = YOLO('yolo11n-seg.pt')
    image = Image.open(image_path)
    
    results = model.predict(image, save=False, conf=0.25, verbose=False)
    
    for i, result in enumerate(results):
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy() 
            
            for j, (mask, box) in enumerate(zip(masks, boxes)):
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            
                mask_image = mask_image.resize(image.size)
                
                segmented = Image.new('RGBA', image.size, (0, 0, 0, 0))
                image_rgba = image.convert('RGBA')
                mask_image = mask_image.convert('L')
                segmented.paste(image_rgba, mask=mask_image)
                
                x1, y1, x2, y2 = box[:4]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cropped_segment = segmented.crop((x1, y1, x2, y2))
                
                cropped_segment = cropped_segment.rotate(-90, expand=True)
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = f'{folder_path}/{base_name}_{i}_{j}.png'
                cropped_segment.save(save_path)


def clean_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, filename))
    

if __name__ == "__main__":
    path = './image_user_seg_yolo'
    clean_folder(path)
    image_folder = 'image_user'
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(supported_formats):
            image_path = os.path.join(image_folder, filename)
            logger.info(f'{filename}')
            segment_image(image_path, path)
