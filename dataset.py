import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class FERPlusDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        # npz đang ở dạng (N, 48, 48, 3). Cần đổi thành (N, 3, 48, 48) cho PyTorch
        self.images = torch.tensor(data['images']).permute(0, 3, 1, 2)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # Trả về tensor nguyên bản kiểu uint8 (chưa augmentation)
        return self.images[idx], self.labels[idx]

def get_dataloaders(config):
    # Khởi tạo dataset
    train_ds = FERPlusDataset(config['data']['train_path'])
    val_ds = FERPlusDataset(config['data']['val_path'])
    
    # DataLoader cơ bản (không tốn CPU tính toán augmentation)
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['data']['batch_size'], 
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['data']['batch_size'], 
        shuffle=False
    )
    
    # Định nghĩa Augmentation chạy trên GPU (KHÔNG dùng occlusion augmentation ở chuẩn này)
    train_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True), # Chuẩn hóa [0, 255] -> [0.0, 1.0]
        v2.Resize((config['data']['img_size'], config['data']['img_size']), antialias=True),
        v2.RandomHorizontalFlip(p=0.5), # Flip
        v2.RandomRotation(degrees=10),  # Xoay nhẹ
        v2.RandomResizedCrop(size=config['data']['img_size'], scale=(0.8, 1.0), antialias=True), # Crop
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((config['data']['img_size'], config['data']['img_size']), antialias=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_loader, val_loader, train_transforms, val_transforms