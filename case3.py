import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import CosineSimilarity
import os

class ObjectDetectorAndMatcher:
    def __init__(self):

        self.model = YOLO('yolo11m.pt')

        self.backbone = self.model.model.model[:9]
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        self.cos = CosineSimilarity(dim=1)
        
    def extract_features(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            features = self.backbone(img_tensor)

        return torch.flatten(features, start_dim=1)

    def detect_and_crop_object(self, image_path):
        results = self.model(image_path)
        result = results[0]
        
        if len(result.boxes) == 0:
            return None
    
        best_box = None
        max_score = -1
        
        for box in result.boxes:
            conf = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            score = conf * area
            
            if score > max_score:
                max_score = score
                best_box = box
        
        if best_box is None:
            return None
            
        image = Image.open(image_path)
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        cropped_image = image.crop((x1, y1, x2, y2))
        
        return cropped_image

    def compute_similarity(self, base_features, user_image_path):
        cropped_object = self.detect_and_crop_object(user_image_path)
        if cropped_object is None:
            return 0.0
            
        user_features = self.extract_features(cropped_object)
        
        similarity = self.cos(base_features, user_features).item()
        return similarity

def main():
    detector = ObjectDetectorAndMatcher()
    total_images = 0
    correct_predictions = 0
    
    base_features_dict = {}
    for base_image_name in os.listdir('base_image'):
        if base_image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            object_name = base_image_name.split('.')[0]
            base_image_path = os.path.join('base_image', base_image_name)
            base_features_dict[object_name] = detector.extract_features(base_image_path)
    
    for user_image_name in os.listdir('user_image'):
        if user_image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
    
            true_label = user_image_name.split('(')[0].strip()
            user_image_path = os.path.join('user_image', user_image_name)
            
            max_similarity = -1
            predicted_label = None
            
            for object_name, base_features in base_features_dict.items():
                similarity = detector.compute_similarity(base_features, user_image_path)
                if similarity > max_similarity:
                    max_similarity = similarity
                    predicted_label = object_name
            
            if true_label == predicted_label:
                print(f"Image: {user_image_name} | {true_label} -> {predicted_label}")
            else:
                print(f"Image: {user_image_name} | {true_label} -> {predicted_label}")
            
            total_images += 1
            if predicted_label == true_label:
                correct_predictions += 1
    
    accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
    print("\nFinal Results:")
    print(f"Total images: {total_images}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
