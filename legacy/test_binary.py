import torch
import torch.optim as optim
import argparse
import re
from utils import test_opt, create_dataloader, custom_transforms, create_dataset
from networks.hog_vit import *
from networks.vit import *
from networks.resnet import *
from networks.hog_cnn import *
from networks.learned_filters import LearnableFilterCNN128

import os
import matplotlib.pyplot as plt
import pathlib
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image

# --- New: Logic to auto-detect latest epoch ---
def get_latest_checkpoint(network, transform, detector):
    ckpt_dir = f'checkpoints/{network}_{transform}_{detector}'
    if not os.path.exists(ckpt_dir):
        print(f"Warning: Directory {ckpt_dir} not found.")
        return None
    
    files = os.listdir(ckpt_dir)
    # Extract numbers from files matching 'ckpt_#.pth'
    epochs = [int(re.findall(r'\d+', f)[0]) for f in files if f.startswith('ckpt_') and f.endswith('.pth')]
    
    if not epochs:
        print(f"Warning: No checkpoints found in {ckpt_dir}")
        return None
        
    return max(epochs)

def test_model(model, network, checkpoint, transform, detector, generator, test_dataset, batch_size=256, device="cuda", num_workers=12):
    if len(test_dataset) == 0:
        raise ValueError(f"The test dataset for '{generator}' is empty. Check its val/ai and val/nature image folders.")
    model.to(device)
    print(f"testing on generator : {generator}")
    print(f'loading ckpt from :checkpoints/{network}_{opt.transform}_{detector}/ckpt_{checkpoint}.pth')
    
    # Logic remains original
    state_dict = torch.load(f'checkpoints/{network}_{opt.transform}_{detector}/ckpt_{checkpoint}.pth', weights_only=True, map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') 
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            # Binary logic preserved
            pred = (torch.sigmoid(outputs) > 0.5).float()
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
    accuracy = correct / len(test_loader.dataset)
    print(f'{generator} : {accuracy:.4f}')
    return accuracy

#----------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    opt = test_opt.TestOptions().parse()
    if opt.network in ("learned_filter_cnn", "fixed_filter_cnn") and opt.transform != "gray_center_224":
        raise ValueError(f"--network {opt.network} requires --transform gray_center_224 (raw grayscale input); gradient transforms already apply the filters.")

    root_dir = f'dataset/genimage/{opt.generator}/val'
    
    # Model selection logic
    match opt.network:
        case "res18_rgb": model = Resnet18()
        case "res18_rgb_unlocked": model = Resnet18_unlocked()
        case "dual_vit": model = DualViT()
        case "dual_cnn": model = DualCNN()
        case "dual_cnn_128": model = DualCNN_128()
        case "learned_filter_cnn": model = LearnableFilterCNN128(trainable=True)
        case "fixed_filter_cnn": model = LearnableFilterCNN128(trainable=False)
        case "dual_cnn_128_6ch": model = DualCNN_128(in_channels=6)
        case "dual_cnn_pool": model = DualCNN_pooling()
        case "res18_2ch": model = Resnet18_2CH()
        case "res18_2ch_unlocked": model = Resnet18_2CH_unlocked()
        case "2ch_vit": model = ViT2CH()
        case "pretrain_vit": model = PretrainViT()
        case "HogHist8D": model = HogHist8D()
        case "hog_cnn": model = HogCNN()
        case "pret_vit":
            model = ViTRGB()
        case "pret_vit_2ch":
            model = ViT2ch()
        case "train_fil":
            model = DualCNN_128_trainFilter()
        case "_": print("Unknown Model")

    # Transform selection logic
    match opt.transform:
        case "hog_224": transform = custom_transforms.hog_224
        case "hog_224_vit": transform = custom_transforms.hog_224_vit
        case "hogfft_224": transform = custom_transforms.hogfft_224
        case "hogfft_256": transform = custom_transforms.hogfft_256
        case "hog_256": transform = custom_transforms.hog_256
        case "hog_hist": transform = custom_transforms.hog_hist
        case "hog_hist_8D": transform = custom_transforms.hog_hist_8D
        case "rgb_224": transform = custom_transforms.rgb_224
        case "HoG_224": transform = custom_transforms.HoG_224
        case "grad_224": transform = custom_transforms.grad_224
        case "gray_224":
            transform = custom_transforms.gray_224
        case "gray_center_224":
            transform = custom_transforms.gray_center_224
        case "raw_grad_224":
            transform = custom_transforms.raw_grad_224
        case "rgb_grad_224":
            transform = custom_transforms.rgb_grad_224
        case "prewitt_grad_224":
            transform = custom_transforms.prewitt_grad_224
        case "scharr_grad_224":
            transform = custom_transforms.scharr_grad_224
        case "dxdy_224":
            transform = custom_transforms.dxdy_224
        case "_": print("Unknown Transform")

    detectors = ["sdv4",	"sdv5",	"adm",	"vqdm",	"biggan",	"glide",	"midjourney",	"wukong"]
    generators = ["sdv4",	"sdv5",	"adm",	"vqdm",	"biggan",	"glide",	"midjourney",	"wukong"]
    
    for gen in generators:
        accuracy = []
        for det in detectors:
            # --- Auto-detect the checkpoint epoch ---
            latest_ckpt = get_latest_checkpoint(opt.network, opt.transform, det)
            
            if latest_ckpt is None:
                continue

            root_dir = f'dataset/genimage/{gen}/val'
            dataset = create_dataset.genImage(root_dir=root_dir, transform=transform)
            
            print(f'dataroot:{root_dir}')
            print(f'dataset length: {len(dataset)}')
            
            # Call test_model with the detected latest_ckpt
            acc = test_model(
                model=model, 
                network=opt.network, 
                checkpoint=latest_ckpt, 
                transform=transform, 
                detector=det, 
                generator=gen, 
                test_dataset=dataset, 
                device=device,
                batch_size=opt.batch_size,
                num_workers=opt.num_workers
            ) 
            accuracy.append(acc)
            
        if accuracy:
            avg = sum(accuracy)/len(accuracy)
            print(f"Average accuracy on {gen} is : {avg:.4f}")
