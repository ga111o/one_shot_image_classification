import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import random

class BaseNetwork(nn.Module):
    def __init__(self, input_shape):
        super(BaseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=10)  # 105x105 -> 96x96
        self.pool1 = nn.MaxPool2d(2)                                # 96x96 -> 48x48
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)             # 48x48 -> 42x42
        self.pool2 = nn.MaxPool2d(2)                               # 42x42 -> 21x21
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4)            # 21x21 -> 18x18
        self.pool3 = nn.MaxPool2d(2)                               # 18x18 -> 9x9
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4)            # 9x9 -> 6x6
        
        # Calculate the size of the flattened features
        self.feature_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(self.feature_size, 1024)
        self.fc2 = nn.Linear(1024, 256)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = self._forward_conv(input)
        return int(output.data.view(batch_size, -1).size(1))

    def _forward_conv(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, input_shape):
        super(SiameseNetwork, self).__init__()
        self.base_network = BaseNetwork(input_shape)

    def forward(self, input1, input2):
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)
        return output1, output2

def euclidean_distance(x1, x2):
    return torch.sqrt(torch.sum(torch.pow(x1 - x2, 2), dim=1, keepdim=True) + 1e-7)

class SiameseDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.labels = [os.path.splitext(f)[0] for f in self.image_files]
        
    def __len__(self):
        return len(self.image_files) * 2  # 각 이미지당 positive/negative 쌍 생성
        
    def __getitem__(self, idx):
        img1_idx = idx // 2
        img1_path = os.path.join(self.image_dir, self.image_files[img1_idx])
        img1_label = self.labels[img1_idx]
        
        # 50% 확률로 같은 라벨(1)이나 다른 라벨(0)의 이미지 선택
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            # 같은 라벨의 이미지 선택
            img2_idx = img1_idx
            label = torch.FloatTensor([1.0])
        else:
            # 다른 라벨의 이미지 선택
            while True:
                img2_idx = random.randint(0, len(self.image_files) - 1)
                if self.labels[img2_idx] != img1_label:
                    break
            label = torch.FloatTensor([0.0])
            
        img2_path = os.path.join(self.image_dir, self.image_files[img2_idx])
        
        img1 = Image.open(img1_path).convert('L')  # grayscale로 변환
        img2 = Image.open(img2_path).convert('L')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, label

# 데이터 전처리를 위한 transform 정의
transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

# 데이터셋과 데이터로더 생성
train_dataset = SiameseDataset('./base_image', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

input_shape = (1, 105, 105)  # Omniglot 이미지 크기
siamese_model = SiameseNetwork(input_shape)

optimizer = optim.Adam(siamese_model.parameters(), lr=0.00006)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(20):
    for batch_idx, (data1, data2, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output1, output2 = siamese_model(data1, data2)
        loss = criterion(euclidean_distance(output1, output2), labels)
        loss.backward()
        optimizer.step()

# 학습 루프 완료 후 테스트 코드 추가
def predict_class(model, test_image, base_images, transform):
    model.eval()
    with torch.no_grad():
        # 테스트 이미지 전처리
        test_tensor = transform(Image.open(test_image).convert('L')).unsqueeze(0)
        
        similarities = {}
        # 각 base 이미지와 비교
        for base_img in os.listdir(base_images):
            base_tensor = transform(Image.open(os.path.join(base_images, base_img)).convert('L')).unsqueeze(0)
            
            # Siamese 네트워크로 임베딩 추출
            output1, output2 = model(test_tensor, base_tensor)
            
            # 유클리드 거리 계산
            distance = euclidean_distance(output1, output2).item()
            class_name = os.path.splitext(base_img)[0]
            similarities[class_name] = distance
        
        # 가장 거리가 가까운(유사한) 클래스 반환
        predicted_class = min(similarities.items(), key=lambda x: x[1])[0]
        return predicted_class, similarities

print("\nTesting predictions:")
test_dir = './test'
correct = 0
total = 0

for test_img in os.listdir(test_dir):
    true_class = ''.join([c for c in test_img.split()[0] if not c.isdigit()])
    
    test_path = os.path.join(test_dir, test_img)
    predicted_class, similarities = predict_class(siamese_model, test_path, './base_image', transform)
    
    predicted_object = ''.join([c for c in predicted_class.split()[0] if not c.isdigit()])
    
    if true_class == predicted_object:
        correct += 1
    total += 1
    
    # print(f"\nTest image: {test_img}")
    print(f"True class: {true_class}")
    print(f"Predicted class: {predicted_class}")
    print("Distances to each class:")
    for class_name, distance in sorted(similarities.items(), key=lambda x: x[1]):
        print(f"{class_name}: {distance:.4f}")

accuracy = (correct / total) * 100
print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct}/{total})")
