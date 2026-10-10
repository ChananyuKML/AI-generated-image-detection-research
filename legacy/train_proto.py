import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import pathlib
import matplotlib.pyplot as plt

# Import your existing utilities and networks
from utils import train_opt, create_dataloader, custom_transforms
from networks.hog_cnn import *
from networks.resnet import Resnet18
from networks.hog_vit import DualViTv2

class RealCenterLoss(nn.Module):
    """
    Research Core: Learns a 'Center' for Real images.
    Pushes Fake images outside a defined margin.
    """
    def __init__(self, embedding_dim=128, margin=1.0):
        super(RealCenterLoss, self).__init__()
        self.center = nn.Parameter(torch.randn(1, embedding_dim))
        self.margin = margin

    def forward(self, embeddings, labels):
        # L2 Distance calculation
        distances = torch.norm(embeddings - self.center, p=2, dim=1)
        
        # Real = 1, Fake = 0
        real_mask = (labels == 1).squeeze()
        fake_mask = (labels == 0).squeeze()
        
        # Minimize distance for Real; Hinge loss for Fake
        loss_real = distances[real_mask].pow(2).mean() if real_mask.any() else 0.0
        loss_fake = F.relu(self.margin - distances[fake_mask]).pow(2).mean() if fake_mask.any() else 0.0
        
        return loss_real + loss_fake

def train_model(model, model_name, transform_name, dataset_name, train_loader, valid_loader, num_epochs=30, lr=1e-4, device="cuda"):
    model.to(device)
    criterion = RealCenterLoss(embedding_dim=128, margin=1.0).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=lr)
    
    out_path = f'checkpoints/proto_{model_name}_{transform_name}_{dataset_name}'
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    
    train_loss_hist, val_loss_hist = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        iteration = 1 # Start iteration counter

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            embeddings = model(data)
            loss = criterion(embeddings, labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # --- Progress Tracking (Like your old code) ---
            train_loss += loss.item() * data.size(0)
            
            # Calculate Training Accuracy based on distance
            with torch.no_grad():
                dist = torch.norm(embeddings - criterion.center, p=2, dim=1).unsqueeze(1)
                pred = (dist < 0.5).float() # Distance < 0.5 means "Real"
                train_correct += (pred == labels).sum().item()

            if iteration % 10 == 1:
                print(f"currently training at iteration : {iteration} / {len(train_loader)}", end="\r", flush=True)
            iteration += 1

        # Epoch Metrics
        train_acc = train_correct / len(train_loader.dataset)
        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}")

        # --- Validation Section ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                embeddings = model(inputs)
                
                loss = criterion(embeddings, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # Validation Accuracy
                dist = torch.norm(embeddings - criterion.center, p=2, dim=1).unsqueeze(1)
                pred = (dist < 0.5).float()
                val_correct += (pred == labels).sum().item()

            val_acc = val_correct / len(valid_loader.dataset)
            avg_val_loss = val_loss / len(valid_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'model_state': model.state_dict(),
                    'center': criterion.center,
                }, f'{out_path}/ckpt_{epoch+1}.pth')
                print(f"? Saved new best model at epoch {epoch+1} with val loss {avg_val_loss:.4f}")

            train_loss_hist.append(avg_train_loss)
            val_loss_hist.append(avg_val_loss)

    return train_loss_hist, val_loss_hist

if __name__ == '__main__':
    # 1. Parse Arguments
    opt = train_opt.TrainOptions().parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataroot = f"{opt.dataroot}/{opt.dataset}"

    # 2. Setup Networks (Using your new DualCNN_Prototypical)
    networks = {
        "dual_proto": DualCNN_Prototypical, # The 128-d output version
        "dual_cnn_128_5ch": DualCNN_128_5ch,
        "res18_rgb": Resnet18,
        "dual_vit": DualViTv2,
        "rgb_ano": RGB_Prototypical,
        "dual_cnn_ano": DualCNN_anomally
    }
    # Lookup and instantiate (Default to dual_proto)
    model = networks.get(opt.network, DualCNN_Prototypical)()

    # 3. Setup Transform
    # You use 'grad_224_5ch' for your 5-channel HOG+RGB setup
    transforms_dict = {
        "grad_224": custom_transforms.grad_224,
        "grad_224_5ch": custom_transforms.grad_224_5ch,
        "rgb_224": custom_transforms.rgb_224
    }
    transform = transforms_dict.get(opt.transform, custom_transforms.grad_224_5ch)

    # 4. Setup Dataloader (Specifically using gen_image)
    print(f"Loading data from: {dataroot} using gen_image loader...")
    train_loader, val_loader = create_dataloader.gen_image(dataroot, opt.batch_size, transform=transform)

    # 5. Execute Training
    print(f"Starting Prototypical Training on {opt.network}...")
    train_hist, val_hist = train_model(
        model=model, 
        model_name=opt.network, 
        transform_name=opt.transform, 
        dataset_name=opt.dataset, 
        train_loader=train_loader, 
        valid_loader=val_loader, 
        num_epochs=opt.epochs, 
        lr=opt.lr, 
        device=device
    )

    # 6. Save Research Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label='Train Loss')
    plt.plot(val_hist, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{opt.network}_{opt.transform} Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'output/graph/{opt.network}_{opt.transform}_{opt.epochs}.png')  # Save the plot
    plt.show()
