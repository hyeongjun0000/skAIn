import os
import time
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# 커스텀 데이터셋
class SkinMultiTaskDataset(Dataset):
    def __init__(self, excel_file, img_dir, transform=None):
        self.df = pd.read_excel(excel_file)
        self.img_dir = img_dir
        self.transform = transform
        self.image_names = self.df['Image_ID'].values
        self.labels = self.df.iloc[:, 2:].values.astype(np.float32)
        self.num_tasks = self.labels.shape[1]
        self.target_names = self.df.columns[2:].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.image_names[idx]).strip()
        
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_name += '.jpg'
            
        class_folder = img_name.split('_')[0] 
        img_path = os.path.join(self.img_dir, class_folder, img_name)
        
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, img_name)
            
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"\n사진을 찾을 수 없음: {img_name}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx])
        return image, label

# 메인
def main():
    print("정상실행")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")

    # 데이터 증강 수정
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("엑셀 데이터셋을 불러옵니다.")
    # 얼굴만 잘라낸 고품질 폴더(dataset_cropped) 연동
    train_dataset = SkinMultiTaskDataset(
        excel_file='dataset/skinalaysis_labeling_train1.xlsx', 
        img_dir='dataset_cropped/train', 
        transform=data_transforms['train']
    )
    valid_dataset = SkinMultiTaskDataset(
        excel_file='dataset/skinanalysis_valid1.xlsx', 
        img_dir='dataset_cropped/valid', 
        transform=data_transforms['valid']
    )

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0),
        'valid': DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)
    }
    
    num_tasks = train_dataset.num_tasks

    # 모델 적용
    print("모델구조 생성")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_tasks)
    model = model.to(device)

    # 정밀 채점 기준, 스케줄러 적용
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    num_epochs = 15
    print("모델 학습을 시작합니다.")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
        scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Loss(Huber): {epoch_loss:.4f} | LR: {current_lr:.6f}")

    print("학습 완료. 통합을 위해 ONNX 포맷으로 변환합니다.")
    model.eval()
    model.to('cpu')
    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
    
    # 이름을 새로 지정
    onnx_path = "skin_model.onnx" 
    
    torch.onnx.export(
        model, dummy_input, onnx_path, 
        input_names=['input'], output_names=['output'], opset_version=11
    )
    print(f"ONNX 파일 생성 완료: {onnx_path}")

if __name__ == '__main__':
    main()