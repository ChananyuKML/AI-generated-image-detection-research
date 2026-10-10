import torch
import torch.optim as optim
import argparse
from utils import test_opt, create_dataloader, custom_transforms, create_dataset
from networks.hog_vit import *
from networks.resnet import *
from networks.hog_cnn import *

import os
import matplotlib.pyplot as plt
import pathlib
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image

def test_model(model, network, checkpoint, transform, detector, generator, test_dataset, batch_size=256, device="cuda"):
    model.to(device)
    print(f"testing on generator : {generator}")
    print(f'loading ckpt from :checkpoints/{network}_{opt.transform}_{detector}/ckpt_{checkpoint}.pth')
    state_dict = torch.load(f'checkpoints/{network}_{opt.transform}_{detector}/ckpt_{checkpoint}.pth', weights_only=True)
    # --- 3. Create a new state_dict without the 'module.' prefix ---
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') # remove `module.`
        new_state_dict[name] = v
    # --- 4. Load the new state_dict into the model --- 
    model.load_state_dict(new_state_dict)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=12)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            pred = (torch.sigmoid(outputs) > 0.5).float()
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / len(test_loader.dataset)
    print(f'{generator} : {accuracy:.4f}')
    return accuracy

#----------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available else 'cpu'
    torch.cuda.empty_cache()
    opt = test_opt.TestOptions().parse()

    root_dir = f'dataset/genimage/{opt.generator}/val'
    match opt.network:
        case "res18_rgb":
            model = Resnet18()
        case "res18_rgb_unlocked":
            model = Resnet18_unlocked()
        case "dual_vit":
            model = DualViT()
        case "dual_cnn":
            model = DualCNN()
        case "dual_cnn_128":
            model = DualCNN_128()
        case "res18_2ch":
            model = Resnet18_2CH()
        case "res18_2ch_unlocked":
            model = Resnet18_2CH_unlocked()
        case "2ch_vit":
            model = ViT2CH()
        case "pretrain_vit":
            model = PretrainViT()
        case "HogHist8D":
            model = HogHist8D()
        case "hog_cnn":
            model = HogCNN()
        case "_":
            print("Unknown Loader")

    match opt.transform:
        case "hog_224":
            transform = custom_transforms.hog_224
        case "hog_224_vit":
            transform = custom_transforms.hog_224_vit
        case "hogfft_224":
            transform = custom_transforms.hogfft_224
        case "hogfft_256":
            transform = custom_transforms.hogfft_256
        case "raw_grad_224":
            transform = custom_transforms.raw_grad_224
        case "hog_256":
            transform = custom_transforms.hog_256
        case "hog_hist":
            transform = custom_transforms.hog_hist
        case "hog_hist_8D":
            transform = custom_transforms.hog_hist_8D
        case "rgb_224":
            transform = custom_transforms.rgb_224
        case "HoG_224":
            transform = custom_transforms.HoG_224
        case "_":
            print("Unknown Loader")

    match opt.loader:
        case "gen_image":
            dataset = create_dataset.genImage(root_dir=root_dir, transform=transform)
        case "image_folder":
            dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        case "_":
            print("Unknown Loader")
    detectors = ["sdv4"]
    generators = ["sdv4_LR_112", "sdv4_LR_64", "sdv4_JPEG_65","sdv4_JPEG_30","sdv4_Blur_3","sdv4_Blur_5",]
    ckpt = [96]
    for gen in generators:
        accuracy = []
        for d, det in enumerate(detectors):
            root_dir = f'dataset/genimage/{gen}/val'
            dataset = create_dataset.genImage(root_dir=root_dir, transform=transform)
            print(f'dataroot:{root_dir}')
            print(f'dataset length: {len(dataset)}')
            acc = test_model(model=model, network=opt.network, checkpoint=ckpt[d], transform=transform, detector=det, generator=gen, test_dataset=dataset, device=device) 
            accuracy.append(acc)
        avg = sum(accuracy)/len(accuracy)
        print(f"Average accuracy on {gen} is : {avg}")
    # checkpoint = opt.checkpoint
    # test_model(model=model, network=opt.network, checkpoint=opt.checkpoint, transform=t:qransform, test_dataset=dataset, device=device)
