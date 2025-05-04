import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler

# ResNet 모델 초기화
model = models.resnet50(pretrained=True)
# 마지막 FC 층 제거하고 특징 추출기로 사용
model = torch.nn.Sequential(*list(model.children())[:-1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 이미지 전처리를 위한 transform 정의
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image):
    # 이미지 전처리
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 특징 추출
    with torch.no_grad():
        features = model(img_tensor)
        # 특징 벡터 형태 변환 (batch_size, features)
        features = features.squeeze().cpu().numpy()
    return features.reshape(1, -1)

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        images.append(img)
        filenames.append(filename)
    return images, filenames

# 기본 이미지 로드 및 특징 추출
base_folder = './base_image'
base_images, base_filenames = load_images_from_folder(base_folder)
base_features = np.vstack([extract_features(img) for img in base_images])

def find_most_similar(input_image_path):
    input_img = Image.open(input_image_path).convert('RGB')
    input_features = extract_features(input_img)
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(input_features, base_features)
    most_similar_index = np.argmax(similarities)
    return most_similar_index, similarities[0, most_similar_index]

def process_all_images(user_images_folder):
    results = []
    count = 0
    total = 0
    for filename in os.listdir(user_images_folder):
        input_image_path = os.path.join(user_images_folder, filename)
        try:
            result_index, similarity_score = find_most_similar(input_image_path)
            # similarity가 0.5 이상인 경우만 결과에 추가
            if similarity_score >= 0.5:
                # 사용자 이미지 파일명은 공백 앞까지만
                user_filename_display = filename.split()[0]
                # 기준 이미지 파일명은 . 앞까지만
                base_filename_display = base_filenames[result_index].split('.')[0]
                
                results.append({
                    'filename': filename,
                    'most_similar_index': result_index,
                    'base_filename': base_filenames[result_index],
                    'similarity_score': similarity_score
                })
                print(f"{user_filename_display} -> {base_filename_display} (similarity: {similarity_score:.4f})")
                if base_filename_display == user_filename_display:
                    count+= 1
                total+=1
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    return results, count, total

user_images_folder = '../seg_user_image'
results, count, total = process_all_images(user_images_folder)
print(f"Total: {total}, Correct: {count}, Accuracy: {count/total:.2f}")