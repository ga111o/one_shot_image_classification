import os
from PIL import Image
import numpy as np
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def segment_image(image_path):
    os.makedirs('./seg', exist_ok=True)
    model = YOLO('yolo11l-seg.pt')
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
                
                # 이미지를 올바른 방향으로 회전
                cropped_segment = cropped_segment.rotate(-90, expand=True)
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = f'./seg_user_image/{base_name}_segment_{i}_{j}.png'
                cropped_segment.save(save_path)


if __name__ == "__main__":
    image_folder = 'user_image'
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(supported_formats):
            image_path = os.path.join(image_folder, filename)
            logger.info(f'Processing {filename}...')
            segment_image(image_path)