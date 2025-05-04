import torch
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(img_tensor)
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

base_folder = './image_base'
base_images, base_filenames = load_images_from_folder(base_folder)
base_features = np.vstack([extract_features(img) for img in base_images])

def find_most_similar(input_image_path):
    input_img = Image.open(input_image_path).convert('RGB')
    input_features = extract_features(input_img)
    
    similarities = cosine_similarity(input_features, base_features)
    most_similar_index = np.argmax(similarities)
    return most_similar_index, similarities[0, most_similar_index]

def process_all_images(user_images_folder):
    results = []
    count = 0
    total = 0
    correct_set = set()
    all_set = set()
    for filename in os.listdir(user_images_folder):
        input_image_path = os.path.join(user_images_folder, filename)
        try:
            result_index, similarity_score = find_most_similar(input_image_path)
            if similarity_score >= 0.7:
                user_filename_display = filename.split()[0]
                base_filename_display = base_filenames[result_index].split()[0]
                
                results.append({
                    'filename': filename,
                    'most_similar_index': result_index,
                    'base_filename': base_filenames[result_index],
                    'similarity_score': similarity_score
                })
                prefix = ' '
                if base_filename_display == user_filename_display:
                    count+= 1
                    prefix = 'o'
                    correct_set.add(base_filename_display)
                print(f"{prefix} {filename} -> {base_filenames[result_index]} ({similarity_score:.4f})")
                total+=1
            all_set.add(base_filename_display)

        except Exception as e:
            print(f"errr {filename}: {str(e)}")
    return results, count, total, correct_set, all_set

user_images_folder = './image_user_seg'
results, count, total, correct_set, all_set = process_all_images(user_images_folder)
print(f"{total} / {count}, {count/total:.2f}")
print(correct_set)
print(all_set)