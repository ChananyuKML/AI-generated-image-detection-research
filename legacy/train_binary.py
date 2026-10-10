import torch.optim as optim
import torch
import torch.nn as nn
import argparse
from utils import train_opt, create_dataloader, custom_transforms, create_dataset
from networks.hog_vit import *
from networks.resnet import *
from networks.hog_cnn import *
import os
import matplotlib.pyplot as plt
import pathlib

def train_model(model, model_name, transform, dataroot, dataset, train_loader, valid_loader, num_epochs=30, lr=1e-4, device="cuda"):
    # Check for Multi-GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"?? Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    # Debug Info
    print(f"Torch Version: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    
    out_path = f'checkpoints/{model_name}_{transform}_{dataset}'
    os.makedirs(out_path, exist_ok=True)
    os.makedirs('output/graph', exist_ok=True)

    train_loss_hist = []
    val_loss_hist = []
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        iteration = 1 # Start iteration counter

        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            pred = (torch.sigmoid(outputs) > 0.5).float()
            train_loss += loss.item() * data.size(0)
            correct += (pred == labels).sum().item()
            
            if iteration % 10 == 1:
                print(f"currently training at iteration : {iteration} / {len(train_loader)}", end="\r", flush=True)
            iteration += 1
            
        acc = correct / len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}, Accuracy: {acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)        
                val_loss += loss.item() * inputs.size(0)
                pred = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (pred == labels).sum().item()
            
            val_acc = val_correct / len(valid_loader.dataset)
            val_loss /= len(valid_loader.dataset)
            print(f"Validation Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            # Save best model logic (Handles DataParallel)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_dict, f'{out_path}/best_model.pth')
                print(f"? New best model saved!")

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
            
    return train_loss_hist, val_loss_hist

if __name__ == '__main__':
    # FIX: Added parentheses to is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = train_opt.TrainOptions().parse()
    dataroot = f"{opt.dataroot}/{opt.dataset}"
    
    # Model Selection
    match opt.network:
        case "res18_rgb": model = Resnet18()
        case "dual_cnn": model = DualCNN() # Your Hybrid Pooling model
        case "dual_cnn_pool": model = DualCNN_pooling()
        case "dual_cnn_128": model = DualCNN_128()
        case "hog_cnn": model = HogCNN()
        case "res18_2ch": model = Resnet18_2CH()
        case _: print("?? Unknown Network Architecture")

    # Transform Selection
    match opt.transform:
        case "rgb_224": transform = custom_transforms.rgb_224
        case "grad_224": transform = custom_transforms.grad_224
        case "dxdy_224": transform = custom_transforms.dxdy_224
        case _: print("?? Unknown Transform")

    # Loader Selection
    match opt.loader:
        case "createByDir": train_loader, val_loader, _ = create_dataloader.createByDir(dataroot, opt.batch_size, transform=transform)
        case "createByDir2": train_loader, val_loader = create_dataloader.createByDir2(dataroot, opt.batch_size, transform=transform)
        case "gen_image": train_loader, val_loader = create_dataloader.gen_image(dataroot, opt.batch_size, transform=transform)
        case _: print("?? Unknown Loader")

    print(f"Initialization complete. Training on: {device}")
    
    train_hist, val_hist = train_model(
        model=model, 
        model_name=opt.network, 
        dataset=opt.dataset, 
        dataroot=dataroot, 
        transform=opt.transform, 
        train_loader=train_loader, 
        valid_loader=val_loader, 
        num_epochs=opt.epochs, 
        lr=opt.lr, 
        device=device
    )
    
    # 6. Save Research Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(train_hist, label='Train Distance Loss')
    plt.plot(val_hist, label='Val Distance Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (L2 Distance from Real Center)')
    plt.title(f'Metric Learning: {opt.network} on {opt.dataset}')
    plt.legend()
    plt.grid(True)
    os.makedirs('output/graph', exist_ok=True)
    plt.savefig(f'output/graph/proto_{opt.network}_{opt.dataset}.png')
    plt.show()