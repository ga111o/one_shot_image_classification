from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('L')
        img = img.resize((256, 256))
        images.append(np.array(img).flatten())
        filenames.append(filename)
    return np.array(images), filenames

base_folder = './image_base'
base_images, base_filenames = load_images_from_folder(base_folder)

scaler = StandardScaler()
base_images_scaled = scaler.fit_transform(base_images)

clf = svm.SVC(kernel='linear', probability=True)
labels = np.arange(len(base_images_scaled))
clf.fit(base_images_scaled, labels)

def find_most_similar(input_image_path):
    input_img = Image.open(input_image_path).convert('L')
    input_img = input_img.resize((256, 256))
    input_img_flattened = np.array(input_img).flatten()
    input_img_scaled = scaler.transform([input_img_flattened])

    base_features = clf.decision_function(base_images_scaled)
    input_features = clf.decision_function(input_img_scaled)

    similarities = cosine_similarity(input_features, base_features)
    most_similar_index = np.argmax(similarities)
    return most_similar_index, similarities[0, most_similar_index]

def process_all_images(user_images_folder):
    results = []
    for filename in os.listdir(user_images_folder):
        input_image_path = os.path.join(user_images_folder, filename)
        try:
            result_index, similarity_score = find_most_similar(input_image_path)
            results.append({
                'filename': filename,
                'most_similar_index': result_index,
                'base_filename': base_filenames[result_index],
                'similarity_score': similarity_score
            })
            print(f"{filename} -> {base_filenames[result_index]} (similarity: {similarity_score:.4f})")
        except Exception as e:
            print(f"errr {filename}: {str(e)}")
    return results

user_images_folder = './image_user_seg'
results = process_all_images(user_images_folder)
