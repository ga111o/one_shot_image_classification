from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)

class TripletDataset(Dataset):
    def __init__(self, folder):
        self.images = []
        self.filenames = []
        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        for filename in os.listdir(folder):
            img = Image.open(os.path.join(folder, filename)).convert('L')
            self.images.append(img)
            self.filenames.append(filename)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anchor = self.transforms(self.images[idx])
        
        # Positive: 같은 이미지에서 약간 변형된 버전
        positive = self.transforms(self.images[idx])
        
        # Negative: 랜덤하게 다른 이미지 선택
        neg_idx = np.random.choice([i for i in range(len(self.images)) if i != idx])
        negative = self.transforms(self.images[neg_idx])
        
        return anchor, positive, negative, self.filenames[idx]

def train_triplet_network(model, train_loader, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    triplet_loss = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (anchor, positive, negative, _) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            
            loss = triplet_loss(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

def find_most_similar(model, input_image_path, base_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 입력 이미지 처리
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    input_img = Image.open(input_image_path).convert('L')
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        input_embedding = model(input_tensor)
        
        best_similarity = -1
        best_index = -1
        
        for i in range(len(base_dataset)):
            base_img, _, _, _ = base_dataset[i]
            base_embedding = model(base_img.unsqueeze(0).to(device))
            
            similarity = torch.cosine_similarity(input_embedding, base_embedding).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = i
    
    return best_index, best_similarity

# 메인 실행 코드
base_folder = './base_image'
base_dataset = TripletDataset(base_folder)
train_loader = DataLoader(base_dataset, batch_size=32, shuffle=True)

model = TripletNetwork()
train_triplet_network(model, train_loader)

def process_all_images(model, user_images_folder, base_dataset):
    results = []
    for filename in os.listdir(user_images_folder):
        input_image_path = os.path.join(user_images_folder, filename)
        try:
            result_index, similarity_score = find_most_similar(model, input_image_path, base_dataset)
            results.append({
                'filename': filename,
                'most_similar_index': result_index,
                'base_filename': base_dataset.filenames[result_index],
                'similarity_score': similarity_score
            })
            print(f"{filename} -> {base_dataset.filenames[result_index]} (similarity: {similarity_score:.4f})")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    return results

user_images_folder = '../seg_user_image'
results = process_all_images(model, user_images_folder, base_dataset)
