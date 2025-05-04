import os
import torch
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import time
from ultralytics import YOLO

class OneShortArtworkClassifier:
    def __init__(self, model_name='resnet18', device=None, detector_model='yolov8n.pt'):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if model_name == 'resnet18':
            self.model = models.resnet18(weights='IMAGENET1K_V1')
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V1')
        elif model_name == 'vit_b_16':
            self.model = models.vit_b_16(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        if 'resnet' in model_name:
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        elif 'vit' in model_name:
            self.model.heads = torch.nn.Identity()
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.detector = YOLO(detector_model)
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.support_embeddings = []
        self.support_labels = []
        self.support_images = []
    
    def add_support_image(self, img_path, label):
        img = Image.open(img_path).convert('RGB')
        self.support_images.append((img, label))
        
        embedding = self.get_embedding(img)
        self.support_embeddings.append(embedding)
        self.support_labels.append(label)
        
        print(f"Added support image for class '{label}'")
    
    def load_support_set(self, support_dir):
        for class_name in os.listdir(support_dir):
            class_dir = os.path.join(support_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_file)
                        self.add_support_image(img_path, class_name)
                        break
    
    def load_support_images_from_files(self, support_dir):
        for img_file in os.listdir(support_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(support_dir, img_file)
                label = os.path.splitext(img_file)[0]
                self.add_support_image(img_path, label)
    
    def get_embedding(self, image):
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(img_tensor).squeeze().cpu().numpy()
            
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def classify_roi(self, roi):
        if len(self.support_embeddings) == 0:
            return "No support images", 0.0, float('inf')
        
        roi_embedding = self.get_embedding(roi)
        
        distances = [np.linalg.norm(roi_embedding - emb) for emb in self.support_embeddings]
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        confidence = 1.0 / (1.0 + min_distance)
        
        return self.support_labels[min_idx], confidence, min_distance
    
    def extract_rois_yolo(self, frame, conf_threshold=0.25):
        results = self.detector(frame, conf=conf_threshold)
        rois = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    rois.append((roi, (x1, y1, x2-x1, y2-y1)))
        
        if len(rois) == 0:
            rois.append((frame, (0, 0, frame.shape[1], frame.shape[0])))
            
        return rois
    
    def extract_rois(self, frame, method='yolo', min_size=224, conf_threshold=0.25):
        if method == 'yolo':
            return self.extract_rois_yolo(frame, conf_threshold)
        
        elif method == 'full_frame':
            return [(frame, (0, 0, frame.shape[1], frame.shape[0]))]
        
        elif method == 'sliding_window':
            rois = []
            h, w = frame.shape[:2]
            
            if h < min_size or w < min_size:
                return [(frame, (0, 0, w, h))]
            
            win_sizes = [min(h, w) // 2, min(h, w) // 1.5, min(h, w)]
            step_size = min_size // 2
            
            for win_size in win_sizes:
                win_size = int(win_size)
                for y in range(0, h - win_size + 1, step_size):
                    for x in range(0, w - win_size + 1, step_size):
                        roi = frame[y:y+win_size, x:x+win_size]
                        rois.append((roi, (x, y, win_size, win_size)))
            
            return rois
        
        else:
            raise ValueError(f"Method {method} not supported")
    
    def process_video(self, video_source=0, roi_method='yolo', confidence_threshold=0.5, display=True):
        if len(self.support_embeddings) == 0:
            print("No support images loaded. Please add support images first.")
            return
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        print(f"Processing video from source: {video_source}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            rois_and_coords = self.extract_rois(frame, method=roi_method)
            rois = [roi for roi, _ in rois_and_coords]
            
            best_confidence = 0
            best_prediction = None
            best_roi_coords = None
            
            if len(rois) > 0:
                imgs = [self.preprocess(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))) for roi in rois]
                imgs_tensor = torch.stack(imgs).to(self.device)
                with torch.no_grad():
                    embeddings = self.model(imgs_tensor).squeeze().cpu().numpy()
                
                if len(rois) == 1:
                    embeddings = embeddings.reshape(1, -1)
                
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                for i, emb in enumerate(embeddings):
                    distances = [np.linalg.norm(emb - s_emb) for s_emb in self.support_embeddings]
                    min_idx = np.argmin(distances)
                    min_distance = distances[min_idx]
                    confidence = 1.0 / (1.0 + min_distance)
                    prediction = self.support_labels[min_idx]
                    coords = rois_and_coords[i][1]
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_prediction = prediction
                        best_roi_coords = coords
            
            process_time = time.time() - start_time
            fps = 1.0 / process_time
            
            if display and best_confidence > confidence_threshold:
                x, y, w, h = best_roi_coords
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.putText(frame, f"{best_prediction}: {best_confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if display:
                cv2.imshow('One-Shot Artwork Classification', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if display:
            cv2.destroyAllWindows()
    
    def classify_image(self, image_path, roi_method='yolo', confidence_threshold=0.5):
        if len(self.support_embeddings) == 0:
            return "No support images", 0.0, None
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return None, 0.0, None
        
        rois_and_coords = self.extract_rois(frame, method=roi_method)
        rois = [roi for roi, _ in rois_and_coords]
        
        best_confidence = 0
        best_prediction = None
        best_roi_coords = None
        
        if len(rois) > 0:
            imgs = [self.preprocess(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))) for roi in rois]
            imgs_tensor = torch.stack(imgs).to(self.device)
            with torch.no_grad():
                embeddings = self.model(imgs_tensor).squeeze().cpu().numpy()
            
            if len(rois) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            for i, emb in enumerate(embeddings):
                distances = [np.linalg.norm(emb - s_emb) for s_emb in self.support_embeddings]
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                confidence = 1.0 / (1.0 + min_distance)
                prediction = self.support_labels[min_idx]
                coords = rois_and_coords[i][1]
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_prediction = prediction
                    best_roi_coords = coords
        
        if best_confidence > confidence_threshold:
            return best_prediction, best_confidence, best_roi_coords
        else:
            return None, best_confidence, best_roi_coords
    
    def process_image_directory(self, image_dir, roi_method='yolo', confidence_threshold=0.5, save_results=False, output_dir=None):
        if len(self.support_embeddings) == 0:
            print("No support images loaded. Please add support images first.")
            return
        
        if save_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for img_file in os.listdir(image_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(image_dir, img_file)
            print(f"Processing image: {img_path}")
            
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Error: Could not load image {img_path}")
                continue
            
            rois_and_coords = self.extract_rois(frame, method=roi_method)
            rois = [roi for roi, _ in rois_and_coords]
            
            best_confidence = 0
            best_prediction = None
            best_roi_coords = None
            
            if len(rois) > 0:
                imgs = [self.preprocess(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))) for roi in rois]
                imgs_tensor = torch.stack(imgs).to(self.device)
                with torch.no_grad():
                    embeddings = self.model(imgs_tensor).squeeze().cpu().numpy()
                
                if len(rois) == 1:
                    embeddings = embeddings.reshape(1, -1)
                
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                for i, emb in enumerate(embeddings):
                    distances = [np.linalg.norm(emb - s_emb) for s_emb in self.support_embeddings]
                    min_idx = np.argmin(distances)
                    min_distance = distances[min_idx]
                    confidence = 1.0 / (1.0 + min_distance)
                    prediction = self.support_labels[min_idx]
                    coords = rois_and_coords[i][1]
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_prediction = prediction
                        best_roi_coords = coords
            
            result = {
                'detected': best_confidence > confidence_threshold,
                'class': best_prediction if best_confidence > confidence_threshold else None,
                'confidence': best_confidence
            }
            results[img_file] = result
            
            print(f"  Result: {result}")
            
            if save_results and output_dir and best_confidence > confidence_threshold:
                x, y, w, h = best_roi_coords
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{best_prediction}: {best_confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                filename, ext = os.path.splitext(img_file)
                confidence_str = f"{best_confidence:.2f}"
                new_filename = f"{filename}_{best_prediction}_{confidence_str}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                cv2.imwrite(output_path, frame)
        
        return results

if __name__ == "__main__":
    classifier = OneShortArtworkClassifier(detector_model='yolov8n.pt')
    
    base_dir = "./image/image_base"
    if os.path.exists(base_dir):
        classifier.load_support_images_from_files(base_dir)
    else:
        print(f"Base image directory '{base_dir}' not found. Please check the path.")
        exit(1)
    
    user_dir = "./image/image_user"
    if os.path.exists(user_dir):
        results = classifier.process_image_directory(
            user_dir, 
            roi_method='yolo', 
            confidence_threshold=0.7,
            save_results=True,
            output_dir="./results"
        )
        
        print("\nClassification Results Summary:")
        for img_name, result in results.items():
            status = "Detected" if result['detected'] else "Not detected"
            if result['detected']:
                print(f"{img_name}: {status} - {result['class']} ({result['confidence']:.2f})")
            else:
                print(f"{img_name}: {status}")
    else:
        print(f"User image directory '{user_dir}' not found. Please check the path.")
