import yaml
import torch
import timm
from torch import nn, optim
from torch.cuda.amp import GradScaler
from dataset import get_dataloaders
from engine import train_one_epoch, evaluate, save_checkpoint

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {device}")

    train_loader, val_loader, train_tf, val_tf = get_dataloaders(config)
    
    # chuyển lên gpu
    model = timm.create_model(
        config['model']['name'], 
        pretrained=config['model']['pretrained'], 
        num_classes=config['data']['num_classes'],
        drop_path_rate=config['train']['drop_path']
    )
    model = model.to(device)
    
    #loss, optimizer và scaler (AMP)
    criterion = nn.CrossEntropyLoss(label_smoothing=config['train']['label_smoothing'])
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['train']['lr_backbone'], 
        weight_decay=config['train']['weight_decay']
    )
    scaler = GradScaler()
    
    print("Bắt đầu Smoke Test 2 epoch...")
    epochs_to_run = 2 # local test
    
    for epoch in range(epochs_to_run):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, train_tf, device)
        val_acc, val_f1 = evaluate(model, val_loader, val_tf, device)
        
        print(f"Epoch [{epoch+1}/{epochs_to_run}] | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Lưu checkpoint 
        save_checkpoint(epoch, model, optimizer, scaler, val_acc, val_f1)
        
if __name__ == '__main__':
    main()