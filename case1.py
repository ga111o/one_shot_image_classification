import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.segmentation import felzenszwalb
from skimage.measure import regionprops
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('object_detection.log'),
        logging.StreamHandler()
    ]
)

def extract_object_features(image_path):
    logging.info(f"Processing image: {image_path}")

    image = cv2.imread(image_path)
    logging.info(f"Image loaded: {image_path}")
    if image is None:
        raise ValueError("image is None.")

    max_dimension = 640 
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
        logging.info(f"Image resized from {width}x{height} to {new_width}x{new_height}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logging.info("Image converted to RGB")

    segments = felzenszwalb(image_rgb, scale=100, sigma=0.5, min_size=50)
    logging.info("Image segmented")

    object_features = []
    logging.info("Starting to extract features from segments")

    for region in regionprops(segments + 1):
        logging.info(f"Processing segment {region.label}")
    
        mask = segments == (region.label - 1)
        logging.info("Segment mask created")
    
        segment = image_rgb.copy()
        segment[~mask] = 0
        logging.info("Segment extracted")
        if region.area > 100: 
            logging.info("Segment area is greater than 100")
        
            gray_segment = cv2.cvtColor(segment, cv2.COLOR_RGB2GRAY)
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray_segment, n_points, radius, method='uniform')
            logging.info("LBP features extracted")
        
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            logging.info("Histogram calculated")
        
            object_features.append({
                'bbox': region.bbox,
                'area': region.area,
                'histogram': hist,
                'segment': segment
            })
            logging.info("Feature stored")
    
    logging.info(f"Found {len(object_features)} valid segments in the image")
    return object_features

def compare_objects(features1, features2, threshold=0.3):
    score = cv2.compareHist(
        np.float32(features1['histogram']),
        np.float32(features2['histogram']),
        cv2.HISTCMP_BHATTACHARYYA
    )
    logging.debug(f"Comparison score: {score} (threshold: {threshold})")
    return score < threshold

def detect_similar_objects(image_path, target_features):
    objects = extract_object_features(image_path)
    similar_objects = []
    
    for obj in objects:
        if compare_objects(obj, target_features):
            similar_objects.append(obj)
    
    return similar_objects

def main():
    logging.info("Starting object detection process")
    
    train_folder = "image_train"
    test_folder = "image_test"

    logging.info("Processing training images...")
    train_features = {} 
    for image_file in os.listdir(train_folder):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        logging.info(f"Processing training image: {image_file}")
    
        label = os.path.splitext(image_file)[0]

        image_path = os.path.join(train_folder, image_file)
        
        try:
            objects = extract_object_features(image_path)
            if objects:
                if label not in train_features:
                    train_features[label] = []
                train_features[label].extend(objects)
                logging.info(f"Successfully extracted features from {image_file} with label: {label}")
        except Exception as e:
            logging.error(f"Error processing training image {image_file}: {str(e)}")
    
    logging.info(f"Completed training phase. Total labels: {len(train_features)}")
    
    logging.info("Starting test image processing...")
    for image_file in os.listdir(test_folder):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        logging.info(f"Processing test image: {image_file}")

        image_path = os.path.join(test_folder, image_file)

        try:
            test_objects = extract_object_features(image_path)
            
            if test_objects:
                logging.info(f"Found {len(test_objects)} objects in test image {image_file}")
                image = cv2.imread(image_path)
                    
                for obj in test_objects:
                    best_score = float('inf')
                    best_label = None

                    logging.info("Starting to compare features")
                    for label, features_list in train_features.items():
                        for train_obj in features_list:
                            score = cv2.compareHist(
                                np.float32(obj['histogram']),
                                np.float32(train_obj['histogram']),
                                cv2.HISTCMP_BHATTACHARYYA
                            )
                            if score < best_score:
                                best_score = score
                                best_label = label
                    
                    if best_label and best_score < 0.3: 
                        y1, x1, y2, x2 = obj['bbox']
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, best_label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                output_path = os.path.join('results', f'detected_{image_file}')
                os.makedirs('results', exist_ok=True)
                cv2.imwrite(output_path, image)
                logging.info(f"Saved detection result to: {output_path}")
                
                cv2.imshow(f"Detection Result - {image_file}", image)
                cv2.waitKey(0)
                
        except Exception as e:
            logging.error(f"Error processing test image {image_file}: {str(e)}")
    
    logging.info("Object detection process completed")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
