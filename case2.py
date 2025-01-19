from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image

def compare_images(img1, img2):
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))
  
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
  
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
  
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

def identify_object(user_image_path, base_images_dir):
    model = YOLO('yolo11m.pt')
  
    user_img = cv2.imread(user_image_path)
      
    results = model(user_img)
    
    best_match = None
    best_similarity = -1
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
          
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                      
            detected_obj = user_img[y1:y2, x1:x2]
                      
            for base_img_name in os.listdir(base_images_dir):
                base_img_path = os.path.join(base_images_dir, base_img_name)
                base_img = cv2.imread(base_img_path)
                
                similarity = compare_images(detected_obj, base_img)                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = os.path.splitext(base_img_name)[0] 
    
    return best_match

if __name__ == "__main__":
    total_images = 0
    correct_predictions = 0
    
    for user_image_name in os.listdir('user_image'):
        if user_image_name.lower().endswith(('.png', '.jpg', '.jpeg')):          
            true_label = user_image_name.split('(')[0].strip()
            
            user_image_path = os.path.join('user_image', user_image_name)
            predicted_label = identify_object(user_image_path, 'base_image/')
                      
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
