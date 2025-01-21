import os
from PIL import Image
import numpy as np
import logging
import torch
from ultralytics import YOLO
from torch.nn.functional import cosine_similarity
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ImageMatcher:
    def __init__(self):
        self.model = YOLO('yolo11n-seg.pt')
        self.backbone = self.model.model.model[:9]
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        self.base_features_cache = {} 
    
    def segment_image(self, image_path):
        os.makedirs('./seg', exist_ok=True)
        
        image = Image.open(image_path)
        
        results = self.model.predict(image, save=False, conf=0.25, verbose=False)
        
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
                    
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    save_path = f'./seg/{base_name}_segment_{i}_{j}.png'
                    cropped_segment.save(save_path)

    def extract_features(self, image_path):
        self.segment_image(image_path)
        
        image = Image.open(image_path).convert('RGB')
        original_features = self._extract_single_features(image)
        
        segment_features = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        seg_dir = './seg'
        
        for seg_file in os.listdir(seg_dir):
            if seg_file.startswith(base_name):
                seg_path = os.path.join(seg_dir, seg_file)
                seg_image = Image.open(seg_path).convert('RGB')
                seg_features = self._extract_single_features(seg_image)
                segment_features.append(seg_features)
        
    
        if not segment_features:
        
            zero_features = torch.zeros_like(original_features)
            return torch.cat([original_features, zero_features], dim=1)
        
    
        if len(segment_features) > 1:
            segment_features = segment_features[:1] 
        
        all_features = [original_features] + segment_features
        combined_features = torch.cat(all_features, dim=1)
        
    
        expected_size = original_features.size(1) * 2
        if combined_features.size(1) < expected_size:
            padding_size = expected_size - combined_features.size(1)
            padding = torch.zeros((combined_features.size(0), padding_size), device=combined_features.device)
            combined_features = torch.cat([combined_features, padding], dim=1)
        
        return combined_features
    
    def _extract_single_features(self, image):
    
        target_size = (224, 224) 
        image = image.resize(target_size)
        
    
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
    
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            self.backbone = self.backbone.cuda()
        
        with torch.no_grad():
        
            features = self.backbone(image_tensor)
        
            if isinstance(features, list):
                features = features[-1]
        
            features = torch.flatten(features, start_dim=1)
        
        return features.cpu()

def resize_and_compare_images():
    logger.info("Starting image comparison process")
    
    matcher = ImageMatcher()
    base_dir = "base_image/"
    user_dir = "user_image/"
    
    logger.info("Preprocessing base images...")
    for base_img_name in os.listdir(base_dir):
        if base_img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_img_path = os.path.join(base_dir, base_img_name)
            matcher.base_features_cache[base_img_name] = matcher.extract_features(base_img_path)
    logger.info("Base image preprocessing completed")
    
    correct = 0
    total = 0
    
    logger.info("Starting user image processing...")
    user_images = sorted([f for f in os.listdir(user_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], 
                        reverse=True)
    logger.info(f"Found {len(user_images)} user images")
        
    for user_img_name in user_images:
        if user_img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            user_img_path = os.path.join(user_dir, user_img_name)
            
            clean_user_name = user_img_name
            if ' (' in clean_user_name:
                clean_user_name = clean_user_name.split(' (')[0] + clean_user_name.split(')')[-1]
            
            max_similarity = 0
            most_similar_img = None
            
            user_features = matcher.extract_features(user_img_path)
            
        
            for base_img_name, base_features in matcher.base_features_cache.items():
                similarity = cosine_similarity(user_features, base_features).item()
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_img = base_img_name
            
            is_same_name = clean_user_name == most_similar_img
            prefix = "o | " if is_same_name else "  | "
            if max_similarity > 0.55:
                logger.info(f"{prefix}Result: {max_similarity:.4f} | {user_img_name} | {most_similar_img}")
                
                total += 1
                if is_same_name:
                    correct += 1

            else:
                logger.info(f"{prefix}Result: {max_similarity:.4f} | {user_img_name} | {most_similar_img}")
                    

    logger.info(f"Accuracy: {correct / total:.2f}")

def clear_dir():
    seg_dir = './seg'
    if os.path.exists(seg_dir):
        for file in os.listdir(seg_dir):
            file_path = os.path.join(seg_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(seg_dir)
    logger.info("Cleared seg directory")

if __name__ == "__main__":
    clear_dir()
    resize_and_compare_images()
