from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from .create_dataset import *

def createByDir(dataroot, batch_size, transform=None, num_workers=12):
    train_dir = f'{dataroot}/train'
    val_dir = f'{dataroot}/valid'
    test_dir = f'{dataroot}/test'
    
    train_dataset = CNNDetectionataset(root_dir=train_dir, transform=transform)
    val_dataset = CNNDetectionataset(root_dir=val_dir, transform=transform)
    test_dataset = CNNDetectionataset(root_dir=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def createByDir2(dataroot, batch_size, transform=None, num_workers=12):
    train_dir = f'{dataroot}/train'
    val_dir = f'{dataroot}/valid'
    
    train_dataset = CNNDetectionataset2(root_dir=train_dir, transform=transform)
    val_dataset = CNNDetectionataset2(root_dir=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def trainSplit(dataroot, batch_size, transform=None, num_workers=12):
    train_dir = f'{dataroot}/train'
    train_dataset = CNNDetectionataset(root_dir=train_dir, transform=transform)
    train_ratio = 0.8
    dataset_size = len(train_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def gen_image(dataroot, batch_size, transform=None, num_workers=12):
    train_dir = f'{dataroot}/train'
    val_dir = f'{dataroot}/val'
    
    train_dataset = genImage(root_dir=train_dir, transform=transform)
    val_dataset = genImage(root_dir=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
