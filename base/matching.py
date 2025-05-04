import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x):
        return self.conv(x).view(x.size(0), -1)

class MatchingNetwork(nn.Module):
    def __init__(self):
        super(MatchingNetwork, self).__init__()
        self.embedding_network = EmbeddingNetwork()
        
    def forward(self, support_set, query):
        support_embeddings = self.embedding_network(support_set)
        query_embedding = self.embedding_network(query)
        
        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(1), support_embeddings.unsqueeze(0), dim=2)
        
        # Apply softmax to get attention weights
        attention = torch.nn.functional.softmax(similarities, dim=1)
        
        return attention

def load_images_from_folder(folder):
    images = []
    filenames = []
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((84, 84)),  # Resize to match the paper's setup
        transforms.ToTensor(),
    ])
    
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img_tensor = transform(img).unsqueeze(0)
        images.append(img_tensor)
        filenames.append(filename)
    return torch.cat(images, dim=0), filenames

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matching_network = MatchingNetwork().to(device)
matching_network.eval()

base_folder = './base_image'
base_images, base_filenames = load_images_from_folder(base_folder)

def find_most_similar(input_image_path):
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
    ])
    
    input_img = Image.open(input_image_path)
    input_tensor = transform(input_img).unsqueeze(0)
    
    with torch.no_grad():
        attention = matching_network(base_images.to(device), input_tensor.to(device))
        
    most_similar_index = torch.argmax(attention).item()
    similarity_score = attention[0, most_similar_index].item()
    return most_similar_index, similarity_score

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
            print(f"Error processing {filename}: {str(e)}")
    return results

user_images_folder = '../seg_user_image'
results = process_all_images(user_images_folder)
