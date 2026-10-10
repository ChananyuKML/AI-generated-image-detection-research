import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
from utils import test_opt, create_dataloader, custom_transforms, create_dataset
from networks.hog_cnn import *
from networks.resnet import *
from networks.hog_vit import *

# ==========================================
# 1. UTILITY FUNCTIONS
# ==========================================

def get_latest_checkpoint(network, transform, detector):
    """Scan directory to find the highest epoch number."""
    ckpt_dir = f'checkpoints/proto_{network}_{transform}_{detector}'
    if not os.path.exists(ckpt_dir):
        return None
    
    files = os.listdir(ckpt_dir)
    epochs = [int(re.findall(r'\d+', f)[0]) for f in files if f.startswith('ckpt_') and f.endswith('.pth')]
    return max(epochs) if epochs else None

def plot_distribution_histogram(distances, labels, gen_name, opt, best_t):
    """
    Plots a histogram and density curve using only Matplotlib.
    """
    plt.figure(figsize=(12, 6))
    
    # Separate distances (Label 1 = Real, Label 0 = Fake)
    real_dist = distances[labels == 1]
    fake_dist = distances[labels == 0]
    
    # Define common binning for both to ensure alignment
    bins = np.linspace(0, 2.5, 60) 

    # 1. Plot Histograms (using 'step' or 'bar' for research clarity)
    plt.hist(real_dist, bins=bins, color="dodgerblue", label="Real Images", 
             alpha=0.4, density=True, edgecolor='blue', histtype='stepfilled')
    
    plt.hist(fake_dist, bins=bins, color="crimson", label=f"Fake ({gen_name})", 
             alpha=0.4, density=True, edgecolor='red', histtype='stepfilled')
    
    # 2. Add Mean lines
    plt.axvline(np.mean(real_dist), color='blue', linestyle='--', linewidth=1.5)
    plt.axvline(np.mean(fake_dist), color='red', linestyle='--', linewidth=1.5)
    
    # 3. Add Optimal Threshold line
    plt.axvline(best_t, color='black', linestyle='-', linewidth=2, 
                label=f'Optimal Threshold: {best_t:.2f}')
    
    # Formatting
    plt.title(f'Distance Distribution: Reality vs {gen_name}', fontsize=14)
    plt.xlabel('L2 Distance to Reality Center', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.2)
    
    # Save the plot
    out_dir = f'output/plots/{opt.network}'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{gen_name}_hist.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 2. CORE TEST ENGINE
# ==========================================

def test_proto_model(model, network, checkpoint, transform_name, detector, generator, test_dataset, device, opt):
    model.to(device)
    ckpt_path = f'checkpoints/proto_{network}_{transform_name}_{detector}/ckpt_{checkpoint}.pth'
    
    print(f"\n?? Testing {generator} | Loading: {ckpt_path}")
    
    # Load combined data (Model + Center)
    checkpoint_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Standardize state_dict
    state_dict = checkpoint_data['model_state']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    reality_center = checkpoint_data['center'].to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=12)
    all_distances, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Scanning {generator}", leave=False):
            inputs = inputs.to(device)
            embeddings = model(inputs) 
            
            # Metric Learning: Distance from learned reality
            dist = torch.norm(embeddings - reality_center, p=2, dim=1)
            
            all_distances.extend(dist.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    # Optimization: Find best threshold for this specific generator
    thresholds = np.linspace(0.0, 3.0, 300)
    best_acc, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (all_distances < t).astype(float) # Dist < Threshold = Real (1)
        acc = (preds == all_labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t

    # Visual Reporting
    plot_distribution_histogram(all_distances, all_labels, generator, opt, best_t)
    
    print(f"? {generator} Result: Acc={best_acc:.4f} | Opt-T={best_t:.2f} | Mean-D={all_distances.mean():.3f}")
    return best_acc

# ==========================================
# 3. EXECUTION
# ==========================================

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt = test_opt.TestOptions().parse()

    # Dynamic Network Selection
    networks = {
        "dual_proto": DualCNN_Prototypical,
        "dual_cnn_128_5ch": DualCNN_128_5ch,
        "hog_cnn": HogCNN,
        "res18_2ch": Resnet18_2CH
    }
    model_class = networks.get(opt.network)
    if not model_class:
        raise ValueError(f"Network {opt.network} not supported.")
    model = model_class()

    # Dynamic Transform Selection
    transforms_dict = {
        "hog_224": custom_transforms.hog_224,
        "grad_224_5ch": custom_transforms.grad_224_5ch,
        "rgb_224": custom_transforms.rgb_224
    }
    transform = transforms_dict.get(opt.transform)

    detectors = ["sdv4"] 
    generators = ["sdv4", "sdv5", "adm", "glide", "biggan", "vqdm", "midjourney", "wukong"]

    for gen in generators:
        results = []
        for det in detectors:
            latest = get_latest_checkpoint(opt.network, opt.transform, det)
            if latest is None: continue
            
            root_dir = f'dataset/genimage/{gen}/val'
            if not os.path.exists(root_dir): continue
            
            ds = create_dataset.genImage(root_dir=root_dir, transform=transform)
            acc = test_proto_model(model, opt.network, latest, opt.transform, det, gen, ds, device, opt)
            results.append(acc)
        
        if results:
            print(f"? Final Avg Acc for {gen}: {np.mean(results):.4f}")