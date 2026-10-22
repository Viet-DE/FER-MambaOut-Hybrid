import torch
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score
import numpy as np
import os

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, transforms, device):
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        # raw data lên gpu
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        #Augmentation trực tiếp trên GPU
        images = transforms(images)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Ép kiểu hỗn hợp (AMP) 
        with autocast(dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, transforms, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            images = transforms(images) # Val transform (resize, normalize)
            
            with autocast(dtype=torch.float16):
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, macro_f1

def save_checkpoint(epoch, model, optimizer, scaler, acc, f1, filepath="checkpoints/last.pth"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'val_acc': acc,
        'val_macro_f1': f1
    }
    torch.save(checkpoint, filepath)