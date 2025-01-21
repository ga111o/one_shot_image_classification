from ultralytics import YOLO
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import CosineSimilarity
import os

class ObjectDetectorAndMatcher:
    def __init__(self):

        self.model = YOLO('yolo11n.pt')

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

    def compute_similarity(self, base_features, user_image_path):
           
        user_features = self.extract_features(user_image_path)
        
        similarity = self.cos(base_features, user_features).item()
        return similarity

def detect_objects(user_image):
    model = YOLO('yolo11n-seg.pt')
    
    results = model(user_image)
    
    result_image = results[0].plot()
    
    detected_objects = []
    for r in results:
        boxes = r.boxes
        masks = r.masks
        
        if masks is not None:
            for box, mask in zip(boxes, masks.data):
                x1, y1, x2, y2 = box.xyxy[0].numpy()
                confidence = box.conf[0].numpy()
                class_id = int(box.cls[0].numpy())
                class_name = model.names[class_id]
                
                mask_array = mask.cpu().numpy()
                
                detected_objects.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'mask': mask_array
                })
    
    return result_image, detected_objects

def process_directory(directory_path):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    base_features_dict = {}
    base_dir = "./base_image/"
    for filename in os.listdir(base_dir):
        if filename.lower().endswith(supported_extensions):
            base_path = os.path.join(base_dir, filename)
            detector = ObjectDetectorAndMatcher()
            base_features = detector.extract_features(base_path)
            base_features_dict[filename] = base_features

    output_dir = "./segmented_objects/"
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            os.remove(file_path)
    else:
        os.makedirs(output_dir)
    
    all_results = {}

    count = 0
    correct_count = 0

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(directory_path, filename)
            user_image = cv2.imread(image_path)
            
            if user_image is not None:
                result_img, objects = detect_objects(user_image)
                
                all_results[filename] = {
                    'result_image': result_img,
                    'detected_objects': objects
                }
                
                for idx, obj in enumerate(objects):
                    mask = obj['mask']
                    mask = cv2.resize(mask, (user_image.shape[1], user_image.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8) * 255  # 이진 마스크로 변환

                    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    
                    masked_img = cv2.bitwise_and(user_image, mask_3channel)
                    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        rect = cv2.minAreaRect(largest_contour)
                        angle = rect[-1]
                        
                        if angle < -45:
                            angle = 90 + angle
                        
                        center = (masked_img.shape[1] // 2, masked_img.shape[0] // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        
                        rotated_masked_img = cv2.warpAffine(masked_img, rotation_matrix, 
                                                          (masked_img.shape[1], masked_img.shape[0]))
                        rotated_mask = cv2.warpAffine(mask, rotation_matrix,
                                                    (mask.shape[1], mask.shape[0]))
                        
                        _, new_objects = detect_objects(rotated_masked_img)
                        
                        if new_objects and len(new_objects) > 0:
                            new_mask = new_objects[0]['mask']
                            new_mask = cv2.resize(new_mask, (rotated_masked_img.shape[1], rotated_masked_img.shape[0]))
                            new_mask = (new_mask > 0.5).astype(np.uint8) * 255
                            
                            new_contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if new_contours:
                                largest_new_contour = max(new_contours, key=cv2.contourArea)
                                
                                epsilon = 0.02 * cv2.arcLength(largest_new_contour, True)
                                approx = cv2.approxPolyDP(largest_new_contour, epsilon, True)
                                
                                if len(approx) >= 4:
                                    rect = cv2.minAreaRect(approx)
                                    box = cv2.boxPoints(rect)
                                    box = np.int32(box)
                                    
                                    width = int(rect[1][0])
                                    height = int(rect[1][1])
                                    if width < height:
                                        width, height = height, width
                                    
                                    dst_points = np.array([
                                        [0, 0],
                                        [width-1, 0],
                                        [width-1, height-1],
                                        [0, height-1]
                                    ], dtype=np.float32)
                                    
                                    src_points = box.astype(np.float32)
                                    src_points = order_points(src_points)
                                    
                                    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                                    final_img = cv2.warpPerspective(rotated_masked_img, matrix, (width, height))
                                    
                                    output_dir = "./segmented_objects/"
                                    os.makedirs(output_dir, exist_ok=True)
                                    
                                    output_path = os.path.join(output_dir, f'{filename[:-4]}_{obj["class"]}_{idx}_transformed.png')
                                    cv2.imwrite(output_path, final_img)
                                    continue

                        coords = cv2.findNonZero(rotated_mask)
                        x, y, w, h = cv2.boundingRect(coords)
                        cropped_masked_img = rotated_masked_img[y:y+h, x:x+w]
                        
                        masked_path = os.path.join(output_dir, f'{filename[:-4]}_{obj["class"]}_{idx}_masked.png')
                        cv2.imwrite(masked_path, cropped_masked_img)

                    max_similarity = 0
                    best_match = None
                    temp_path = os.path.join(output_dir, "temp.png")
                    
                    if 'final_img' in locals():
                        cv2.imwrite(temp_path, final_img)
                    else:
                        cv2.imwrite(temp_path, cropped_masked_img)

                    for base_name, base_feat in base_features_dict.items():
                        detector = ObjectDetectorAndMatcher()
                        similarity = detector.compute_similarity(base_feat, temp_path)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match = base_name

                    print(f"{max_similarity}\t|{best_match}\t|{filename}")
                    count += 1
                    if best_match[:2] == filename[:2]:
                        correct_count += 1

                    if max_similarity > 0.7:
                        if 'final_img' in locals():
                            output_path = os.path.join(output_dir, 
                                f'{filename[:-4]}_{obj["class"]}_{idx}_transformed_sim_{max_similarity:.2f}.png')
                            cv2.imwrite(output_path, final_img)
                        else:
                            masked_path = os.path.join(output_dir, 
                                f'{filename[:-4]}_{obj["class"]}_{idx}_masked_sim_{max_similarity:.2f}.png')
                            cv2.imwrite(masked_path, cropped_masked_img)

                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    print(f"count: {count}\t|correct_count: {correct_count}\t|accuracy: {correct_count/count}")

    return all_results

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상
    rect[2] = pts[np.argmax(s)]  # 우하
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상
    rect[3] = pts[np.argmax(diff)]  # 좌하
    
    return rect

results = process_directory('./user_image/')
