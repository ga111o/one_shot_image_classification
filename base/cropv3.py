import os
from PIL import Image
import numpy as np
import logging
from ultralytics import YOLO
import cv2

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def segment_artwork(image_path, folder_path):
    # 이미지 읽기 및 그레이스케일로 변환
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    height, width = image.shape[:2]
    new_width = 640
    new_height = int(height * (new_width / width))
    
    # 이미지와 그레이스케일 이미지 모두 리사이즈
    image = cv2.resize(image, (new_width, new_height))
    gray = cv2.resize(gray, (new_width, new_height))
    
    # Canny 엣지 검출
    edges = cv2.Canny(gray, 50, 150)
    
    # 적응형 임계값 처리로 이진화 (파라미터 조정)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 5
    )
        
    # 엣지와 이진화 이미지 결합
    combined = cv2.bitwise_or(edges, binary)
    
    # 노이즈 제거
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        processed_count = 0
        contour_index = 0
        
        while processed_count < 5 and contour_index < len(sorted_contours):
            current_contour = sorted_contours[contour_index]
            
            if contour_index > 0:
                area_current = cv2.contourArea(current_contour)
                area_first = cv2.contourArea(sorted_contours[0])
                area_diff_ratio = abs(area_first - area_current) / area_first
                if area_diff_ratio > 0.2:
                    break
            
            epsilon = 0.02 * cv2.arcLength(current_contour, True)
            approx = cv2.approxPolyDP(current_contour, epsilon, True)
            
            if len(approx) == 4:
                pts = np.float32(approx.reshape(4, 2))
                rect = np.zeros((4, 2), dtype="float32")
                
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                
                widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
                widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                
                heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
                heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")
                
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
                
                # HSV 변환 및 마스킹 부분 제거하고 그레이스케일 처리로 대체
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                _, warped_mask = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel)
                warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_OPEN, kernel)
                
                warped_contours, _ = cv2.findContours(warped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if warped_contours:
                    largest_warped_contour = max(warped_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_warped_contour)
                    artwork = warped[y:y+h, x:x+w]
                    
                    aspect_ratio = w / h if h != 0 else float('inf')
                    if 1/3 <= aspect_ratio <= 3:
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        save_path = f'{folder_path}/{base_name}_{processed_count+1}.png'
                        cv2.imwrite(save_path, artwork)
                        processed_count += 1
            
                else:
                    artwork = warped
            
                    aspect_ratio = artwork.shape[1] / artwork.shape[0]
                    if 1/3 <= aspect_ratio <= 3:
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        save_path = f'{folder_path}/{base_name}_{processed_count+1}.png'
                        cv2.imwrite(save_path, artwork)
                        processed_count += 1
            else:
                x, y, w, h = cv2.boundingRect(current_contour)
                artwork = image[y:y+h, x:x+w]
        
                aspect_ratio = w / h if h != 0 else float('inf')
                if 1/3 <= aspect_ratio <= 3:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    save_path = f'{folder_path}/{base_name}_{processed_count+1}.png'
                    cv2.imwrite(save_path, artwork)
                    processed_count += 1
            
            contour_index += 1
    else:
        x, y, w, h = cv2.boundingRect(sorted_contours[0])
        artwork = image[y:y+h, x:x+w]
        
        aspect_ratio = w / h if h != 0 else float('inf')
        if 1/3 <= aspect_ratio <= 3:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = f'{folder_path}/{base_name}_1.png'
            cv2.imwrite(save_path, artwork)

def clean_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, filename))
    

if __name__ == "__main__":
    path = './image_user_segv2'
    clean_folder(path)
    image_folder = 'image_user'
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(supported_formats):
            image_path = os.path.join(image_folder, filename)
            logger.info(f'{filename}')
            segment_artwork(image_path, path)