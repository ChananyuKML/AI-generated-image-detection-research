import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import os
from utils import test_opt, create_dataloader, custom_transforms, create_dataset
from networks.hog_cnn import *
from networks.resnet import *
from networks.hog_vit import *
import numpy as np
import os
import re

def get_latest_checkpoint(network, transform, detector):
    ckpt_dir = f'checkpoints/proto_{network}_{transform}_{detector}'
    if not os.path.exists(ckpt_dir):
        print(f"?? Warning: Directory {ckpt_dir} not found.")
        return None
    
    # Find all files that look like 'ckpt_#.pth'
    files = os.listdir(ckpt_dir)
    ckpt_files = [f for f in files if f.startswith('ckpt_') and f.endswith('.pth')]
    
    if not ckpt_files:
        print(f"?? Warning: No checkpoints found in {ckpt_dir}")
        return None

    # Extract the numbers using regex and find the max
    epochs = [int(re.findall(r'\d+', f)[0]) for f in ckpt_files]
    latest_epoch = max(epochs)
    
    return latest_epoch
    
def test_proto_model(model, network, checkpoint, transform_name, detector, generator, test_dataset, device="cuda"):
    model.to(device)
    
    # --- 1. Load the Prototype Checkpoint ---
    # Path matches your train_proto.py saving logic
    ckpt_path = f'checkpoints/proto_{network}_{transform_name}_{detector}/ckpt_{checkpoint}.pth'
    print(f"\nTesting on generator: {generator}")
    print(f"Loading Proto-Ckpt from: {ckpt_path}")
    
    checkpoint_data = torch.load(ckpt_path, weights_only=False, map_location=device)
    
    # Extract model state and the learned Reality Center
    state_dict = checkpoint_data['model_state']
    reality_center = checkpoint_data['center'].to(device) # Shape: [1, 128]

    # --- 2. Clean State Dict (Standard DataParallel fix) ---
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=12)
    
    # Lists to store results for threshold optimization
    all_distances = []
    all_labels = []

    print(f"Gathering embeddings for {generator}...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # labels: 1 for Real, 0 for Fake
            
            embeddings = model(inputs) 
            # Calculate L2 distance to the learned center
            dist = torch.norm(embeddings - reality_center, p=2, dim=1)
            
            all_distances.extend(dist.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Convert to numpy for fast calculation
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    # --- 2. THE OPTIMIZATION BLOCK ---
    # We test 200 different thresholds between distance 0 and 3.0
    thresholds = np.linspace(0.0, 3.0, 200)
    best_acc = 0.0
    best_t = 0.5 # Default start

    for t in thresholds:
        # If distance < t, predict Real (1), else Fake (0)
        preds = (all_distances < t).astype(float)
        acc = (preds == all_labels).mean()
        
        if acc > best_acc:
            best_acc = acc
            best_t = t

    # --- 3. Final Output ---
    print(f"Results for {generator}:")
    print(f"  - Fixed Threshold (0.5) Acc : {((all_distances < 0.5).astype(float) == all_labels).mean():.4f}")
    print(f"  - OPTIMAL Threshold ({best_t:.2f}) Acc: {best_acc:.4f}")
    print(f"  - Avg Distance to Center    : {all_distances.mean():.4f}")
    
    return best_acc

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    opt = test_opt.TestOptions().parse()

    # --- 5. Network Selection (Ensure these match your 128-d versions) ---
    networks = {
        "dual_proto": DualCNN_Prototypical,
        "dual_cnn_128_5ch": DualCNN_128_5ch,
        "hog_cnn": HogCNN
    }
    model = networks.get(opt.network, DualCNN_Prototypical)()

    # --- 6. Transform Selection ---
    transforms_dict = {
        "hog_224": custom_transforms.hog_224,
        "grad_224_5ch": custom_transforms.grad_224_5ch,
        "rgb_224": custom_transforms.rgb_224
    }
    transform = transforms_dict.get(opt.transform, custom_transforms.grad_224_5ch)

    detectors = ["sdv4"] # You can add more here later, like ["progan", "stylegan"]
    generators = ["sdv4", "sdv5", "adm", "glide", "biggan", "vqdm", "midjourney", "wukong"]
#    generators = ["sdv4_LR_112", "sdv4_LR_64", "sdv4_JPEG_65", "sdv4_JPEG_30", "sdv4_Blur_3", "sdv4_Blur_5"]

    for gen in generators:
        accuracy_list = []
        for det in detectors:
            # --- AUTO-DETECT LATEST EPOCH ---
            latest_epoch = get_latest_checkpoint(opt.network, opt.transform, det)
            
            if latest_epoch is None:
                continue # Skip if no checkpoint found
                
            root_dir = f'dataset/genimage/{gen}/val'
            dataset = create_dataset.genImage(root_dir=root_dir, transform=transform)
            
            # Pass the latest_epoch automatically
            acc = test_proto_model(
                model=model, 
                network=opt.network, 
                checkpoint=latest_epoch, 
                transform_name=opt.transform, 
                detector=det, 
                generator=gen, 
                test_dataset=dataset, 
                device=device
            )
            accuracy_list.append(acc)
        
        if accuracy_list:
            avg = sum(accuracy_list) / len(accuracy_list)
            print(f"Final Average accuracy on {gen}: {avg:.4f}")